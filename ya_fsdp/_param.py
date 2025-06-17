import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, List, Optional, cast

import torch
import torch.nn as nn
from torch._logging import warning_once
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor.device_mesh import _mesh_resources
from torch.distributed.tensor.placement_types import Placement

from ._api import MixedPrecisionPolicy
from ._common import FSDPMeshInfo, TrainingState, _raise_assert_with_print
from ._tensor import YaFSDPDTensor, YaFSDPDTensorSpec, YaFSDPShard, _YaFSDPStridedShard

logger = logging.getLogger("ya_fsdp")


class ShardedState(Enum):
    """
    - ``SHARDED``: The sharded parameter is registered to the module. It is the
      only contributor to parameter memory.
    - ``UNSHARDED``: The unsharded parameter is registered to the module. Both
      it and the sharded parameter contribute to parameter memory.
    """

    SHARDED = auto()
    UNSHARDED = auto()


@dataclass
class ParamModuleInfo:
    """
    For a parameter, this stores the module and the parameter name to be able
    to do a parameter swap via ``setattr(module, param_name, ...)`` or to get
    the parameter via ``getattr(module, param_name)``. We additionally save
    shared modules and shared parameter names to update them accordingly.
    """

    # Parameter names are unprefixed, e.g. "weight", not "lin.weight"
    module: nn.Module
    param_name: str
    shared_modules: List[nn.Module] = field(default_factory=list)
    shared_param_names: List[str] = field(default_factory=list)


@dataclass
class ExtensionsData:
    # User-defined metadata passed from pre to post-all-gather
    all_gather_metadata: Optional[Any] = None

    def clear(self):
        self.all_gather_metadata = None


class YaFSDPParam:
    """
    This class manages a parameter with YaFSDP applied.
    """

    orig_dtype: torch.dtype
    param_dtype: Optional[torch.dtype]
    reduce_dtype: Optional[torch.dtype]
    _orig_size: torch.Size
    _orig_numel: int
    sharded_param: nn.Parameter
    _sharded_param_grad: Optional[torch.Tensor]
    _all_gather_input: Optional[torch.Tensor]
    _unsharded_param: nn.Parameter
    _unsharded_param_grad: Optional[torch.Tensor]
    _all_gather_output: torch.Tensor
    _unsharded_params_with_post_backward_hook: List[nn.Parameter]

    def __init__(
        self,
        param: nn.Parameter,
        module_info: ParamModuleInfo,
        mesh_info: FSDPMeshInfo,
        post_forward_mesh_info: Optional[FSDPMeshInfo],
        device: torch.device,
    ):
        self._module_info: ParamModuleInfo = module_info
        self.mesh_info = mesh_info
        self.post_forward_mesh_info = post_forward_mesh_info
        self.device = device

        if param.device != device:
            raise AssertionError(f"Expects the parameter to already be moved to device {device} but got {param.device}")
        if not param.is_contiguous():
            raise NotImplementedError(
                f"YaFSDP does not support non-contiguous parameters yet: {param.shape=} {param.stride()=}"
            )
        self.is_dtensor = isinstance(param, DTensor)
        if self.is_dtensor:
            self._tp_spec = cast(DTensor, param)._spec
            dp_mesh, tp_mesh = self.mesh_info.mesh, self._tp_spec.mesh
            dp_global_mesh = _mesh_resources.get_root_mesh(dp_mesh)
            tp_global_mesh = _mesh_resources.get_root_mesh(tp_mesh)
            if dp_global_mesh != tp_global_mesh or (dp_global_mesh is None or tp_global_mesh is None):
                raise AssertionError(
                    "YaFSDP requires the DP and TP mesh to have the same parent mesh but got: \n"
                    f"DP's global mesh: {dp_global_mesh}\nTP's global mesh: {tp_global_mesh}"
                )
            name_dims_error = "YaFSDP requires named DeviceMesh dims for ND parallelism"
            assert dp_mesh.mesh_dim_names is not None, name_dims_error
            assert tp_mesh.mesh_dim_names is not None, name_dims_error
            submesh_names = dp_mesh.mesh_dim_names + tp_mesh.mesh_dim_names
            self._spmd_mesh = dp_global_mesh[submesh_names]
            if len(self._tp_spec.placements) != 1:
                raise NotImplementedError(f"YaFSDP only supports 1D TP, not {self._tp_spec.placements}")
            split_factor = self._tp_spec.num_shards_map[0]
            assert self._spmd_mesh.ndim == 2, f"_spmd_mesh.ndim can only be 2 but got {self._spmd_mesh.ndim}."
            self._spmd_placements: tuple[Placement, ...]
            dp_shard_tp_placement = (
                (_YaFSDPStridedShard(split_factor=split_factor) if split_factor > 1 else YaFSDPShard()),
                self._tp_spec.placements[0],
            )
            self._spmd_placements = dp_shard_tp_placement
            self._sharding_spec = DTensorSpec(
                self._spmd_mesh,
                self._spmd_placements,
                tensor_meta=self._tp_spec.tensor_meta,
            )
            if split_factor > 1:  # FSDP has strided sharding on tensor dim 0
                num_shards = self._sharding_spec.num_shards_map[0]
                tensor_size_dim_0 = self._sharding_spec.shape[0]
                if tensor_size_dim_0 % num_shards != 0:
                    raise NotImplementedError(
                        "YaFSDP+TP sharding does not support uneven sharding for now: "
                        f"tensor dim 0 has size {tensor_size_dim_0} which cannot be "
                        f"evenly sharded into {num_shards} shards."
                    )
            param_data = cast(DTensor, param)._local_tensor
        else:
            self._spmd_mesh = self.mesh_info.mesh
            self._spmd_placements = (YaFSDPShard(),)
            self._sharding_spec = DTensorSpec(
                self._spmd_mesh,
                self._spmd_placements,
                tensor_meta=TensorMeta(param.size(), param.stride(), param.dtype),
            )
            param_data = param.data
        assert param_data.is_contiguous(), f"{param_data.shape=} {param_data.stride()=}"
        self._orig_size = param_data.size()
        self._orig_numel = param_data.numel()
        self.param_data = param_data
        self._requires_grad = param.requires_grad

        self._unsharded_params_with_post_backward_hook = []
        self._param_fqn: Optional[str] = None  # prefixed from root module

    @torch.no_grad()
    def _init_sharded_param(
        self,
        sharded_param_data: torch.Tensor,
        sharded_param_grad: Optional[torch.Tensor],
        global_offset: int,
    ):
        sharded_param = self.param_data.new_zeros((sharded_param_data.numel(),), dtype=sharded_param_data.dtype)
        if hasattr(sharded_param, "__tensor_flatten__"):
            inner_tensors = [getattr(sharded_param, attr_name) for attr_name in sharded_param.__tensor_flatten__()[0]]
        else:
            inner_tensors = [sharded_param]
        torch.utils.swap_tensors(inner_tensors[0], sharded_param_data)
        self._sharding_spec = YaFSDPDTensorSpec(
            self._sharding_spec.mesh,
            self._sharding_spec.placements,
            tensor_meta=self._sharding_spec.tensor_meta,
            local_numel=sharded_param.numel(),
            global_offset=global_offset,
        )
        self.sharded_param = nn.Parameter(
            YaFSDPDTensor(
                sharded_param,
                self._sharding_spec,
                requires_grad=False,
            ),
            requires_grad=self._requires_grad,
        )
        self._setattr_on_modules(self.sharded_param)
        self.sharded_state = ShardedState.SHARDED
        self._init_extensions()
        if self._requires_grad:
            assert sharded_param_grad is not None
            self._sharded_param_grad = YaFSDPDTensor(
                sharded_param_grad,
                self._sharding_spec,
                requires_grad=False,
            )
        else:
            self._sharded_param_grad = None
        del self.param_data

    def set_sharded_param_grad(self):
        self.sharded_param.grad = self._sharded_param_grad

    def init_dtype_attrs(self, mp_policy: MixedPrecisionPolicy):
        param_dtype, reduce_dtype = (mp_policy.param_dtype, mp_policy.reduce_dtype)
        self.orig_dtype = self.param_data.dtype
        # Clamp `reduce_dtype` to `None` if no casting is required: since
        # gradients are computed in `param_dtype`, if `reduce_dtype` matches,
        # then we do not need extra casting
        if reduce_dtype == param_dtype:
            reduce_dtype = None
        # Clamp `param_dtype` to `None` if no casting is required
        if param_dtype == self.orig_dtype:
            param_dtype = None
        self.param_dtype = param_dtype
        self.reduce_dtype = reduce_dtype
        # None indicates that the mixed precision is not enabled

    def _init_extensions(self) -> None:
        inner_tensor = self._sharded_local_tensor
        has_fsdp_pre_all_gather = hasattr(inner_tensor, "fsdp_pre_all_gather")
        has_fsdp_post_all_gather = hasattr(inner_tensor, "fsdp_post_all_gather")
        if has_fsdp_pre_all_gather != has_fsdp_post_all_gather:
            raise AssertionError(
                "Both fsdp_pre_all_gather and fsdp_post_all_gather should be defined "
                f"if using all-gather extensions: {inner_tensor}"
            )
        if has_fsdp_pre_all_gather:
            self._extensions_data = ExtensionsData()

    def init_all_gather_input(self, all_gather_input: torch.Tensor):
        self._all_gather_input = all_gather_input

    def init_all_gather_output(self, all_gather_output: torch.Tensor, unsharded_param_grad: Optional[torch.Tensor]):
        self._all_gather_output = all_gather_output.view(self._orig_size)
        if self.sharded_param.requires_grad:
            assert unsharded_param_grad is not None
            unsharded_param_grad = unsharded_param_grad.view(self._orig_size)
            if self.is_dtensor:
                unsharded_param_grad = DTensor(unsharded_param_grad, self._tp_spec, requires_grad=False)
            self._unsharded_param_grad = unsharded_param_grad
        else:
            self._unsharded_param_grad = None

    def init_unsharded_param(self):
        inner_tensor = self._sharded_local_tensor
        if hasattr(inner_tensor, "fsdp_post_all_gather"):
            unsharded_param, _ = inner_tensor.fsdp_post_all_gather(
                (self._all_gather_output,),
                self._extensions_data.all_gather_metadata,
                self.param_dtype or self.orig_dtype,
            )
        else:
            unsharded_param = self._all_gather_output
        if self.is_dtensor:
            unsharded_param = DTensor(unsharded_param, self._tp_spec, requires_grad=False)
        self._unsharded_param = nn.Parameter(
            unsharded_param,
            requires_grad=self.sharded_param.requires_grad,
        )

    def to_sharded(self, training_state: Optional[TrainingState] = None) -> None:
        self._setattr_on_modules(self.sharded_param)
        self.sharded_state = ShardedState.SHARDED

    def to_unsharded(self, training_state: Optional[TrainingState] = None) -> None:
        # Assume that the data has been allocated and all-gathered
        if self.sharded_param.requires_grad and (
            (training_state == TrainingState.FORWARD and torch.is_grad_enabled())
            or training_state == TrainingState.PRE_BACKWARD
        ):
            unsharded_param = self._unsharded_params_with_post_backward_hook[-1]
        else:
            unsharded_param = self._unsharded_param
        self._setattr_on_modules(unsharded_param)
        self.sharded_state = ShardedState.UNSHARDED

    def register_unsharded_param_with_post_backward_hook(self, unsharded_param: nn.Parameter):
        unsharded_param = _PatchGradAccumulation.apply(self, unsharded_param)
        unsharded_param._is_param = True
        self._unsharded_params_with_post_backward_hook.append(cast(nn.Parameter, unsharded_param))

    def _setattr_on_modules(self, param: nn.Parameter) -> None:
        unsafe_setattr_param(self._module_info.module, self._module_info.param_name, param)
        for shared_module, shared_param_name in zip(
            self._module_info.shared_modules, self._module_info.shared_param_names
        ):
            unsafe_setattr_param(shared_module, shared_param_name, param)

    def set_all_gather_input(self):
        self._assert_in_states(ShardedState.SHARDED)
        if hasattr(self._sharded_local_tensor, "fsdp_pre_all_gather"):
            sharded_local_tensor = self._sharded_local_tensor
            (
                all_gather_inputs,
                self._extensions_data.all_gather_metadata,
            ) = sharded_local_tensor.fsdp_pre_all_gather(self.shard_mesh_from_root)
            assert len(all_gather_inputs) == 1
            all_gather_input = next(iter(all_gather_inputs))
            assert self._all_gather_input is not None
            self._all_gather_input.copy_(all_gather_input)

    @property
    def _sharded_local_tensor(self) -> torch.Tensor:
        return cast(DTensor, self.sharded_param)._local_tensor

    @property
    def shard_mesh(self):
        mesh = self.mesh_info.mesh
        if mesh.ndim == 1:
            return mesh
        elif mesh.ndim == 2:
            assert mesh.mesh_dim_names is not None
            return mesh[mesh.mesh_dim_names[-1]]
        raise ValueError(f"Invalid mesh: {mesh}")

    @property
    def shard_mesh_from_root(self):
        mesh = self.mesh_info.mesh

        if mesh.ndim == 1:
            return mesh
        else:
            assert mesh.mesh_dim_names is not None
            shard_dim_name = mesh.mesh_dim_names[-1]

            root_mesh = _mesh_resources.get_root_mesh(mesh)
            return root_mesh[shard_dim_name]

    def _assert_in_states(self, *states: ShardedState) -> None:
        if self.sharded_state not in states:
            _raise_assert_with_print(f"Expects to be in one of {states}, not {self.sharded_state}")

    def __repr__(self):
        return f"YaFSDPParam(fqn={self._param_fqn}, orig_size={self._orig_size})"


# NOTE: These bypass `nn.Module.__setattr__` checks, which incur non-trivial
# CPU overhead, if the module did not override it. For FSDP, we know we do not
# need those checks when transitioning between sharded/unsharded parameters.
# NOTE: With YaFSDP we can't actually use `nn.Module.__setattr__` because
# unsharded_params have a grad_fn during forward, which is prohibited by
# `nn.Module.__setattr__` checks.
def unsafe_setattr_param(module: nn.Module, param_name: str, param: nn.Parameter) -> None:
    if getattr(module.__setattr__, "__func__", None) is not nn.Module.__setattr__:
        msg = f"{module.__class__} defines a custom __setattr__ which YaFSDP does not support."
        warning_once(logger, msg, stacklevel=2)
    module._parameters[param_name] = param


class _PatchGradAccumulation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fsdp_param: YaFSDPParam, unsharded_param: nn.Parameter):
        ctx.fsdp_param = fsdp_param
        ctx.set_materialize_grads(False)
        return unsharded_param

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        fsdp_param: YaFSDPParam = ctx.fsdp_param
        if grad is None:
            return None, None
        if fsdp_param._unsharded_param.grad is None:
            fsdp_param._unsharded_param.grad = fsdp_param._unsharded_param_grad
            fsdp_param._unsharded_param.grad.copy_(grad)
        else:
            fsdp_param._unsharded_param.grad += grad
        return None, None
