import gc
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, cast

import torch
import torch.nn as nn
from torch._logging import warning_once
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor.device_mesh import _mesh_resources
from torch.distributed.tensor.placement_types import Placement, Shard, _StridedShard

from ._api import MixedPrecisionPolicy
from ._common import FSDPMeshInfo, HSDPMeshInfo, TrainingState

logger = logging.getLogger("ya_fsdp")


class ShardedState(Enum):
    """
    - ``SHARDED``: The sharded parameter is registered to the module. It is the
      only contributor to parameter memory.
    - ``SHARDED_POST_FORWARD``: The unsharded parameter is resharded to a
      smaller world size. Since this data should not be used for computation,
      we do not register it to the module. Users should reshard the module
      before any in-place modifications. Both it and the sharded parameter
      contribute to parameter memory.
    - ``UNSHARDED``: The unsharded parameter is registered to the module. Both
      it and the sharded parameter contribute to parameter memory.
    """

    SHARDED = auto()
    SHARDED_POST_FORWARD = auto()
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
    shared_modules: list[nn.Module] = field(default_factory=list)
    shared_param_names: list[str] = field(default_factory=list)


class YaFSDPParam:
    """
    This class manages a parameter with YaFSDP applied.
    """

    _orig_size: torch.Size
    _orig_numel: int
    sharded_param: nn.Parameter
    _sharded_param_grad: torch.Tensor
    _unsharded_param: nn.Parameter
    _unsharded_param_grad: torch.Tensor
    _unsharded_params_for_backward: list[nn.Parameter]

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
                (_StridedShard(0, split_factor=split_factor) if split_factor > 1 else Shard(0)),
                self._tp_spec.placements[0],
            )
            if self._spmd_mesh.ndim == 2:
                self._spmd_placements = dp_shard_tp_placement
            else:
                raise NotImplementedError("YaFSDP doesn't support HSDP yet.")
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
            if isinstance(self.mesh_info, HSDPMeshInfo):
                raise NotImplementedError("YaFSDP doesn't support HSDP yet.")
            else:
                self._spmd_placements = (Shard(0),)
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

        self.sharded_param = None
        self._sharded_param_grad = None

        self._unsharded_param = None
        self._unsharded_param_grad = None
        self._unsharded_params_for_backward = []
        self._param_fqn: Optional[str] = None  # prefixed from root module

    @torch.no_grad()
    def init_sharded_param(
        self,
        sharded_param: nn.Parameter,
        sharded_param_grad: Optional[torch.Tensor],
    ):
        self.sharded_param = nn.Parameter(
            DTensor(
                sharded_param,
                self._sharding_spec,
                requires_grad=False,
            ),
            requires_grad=self._requires_grad,
        )
        if self._requires_grad:
            assert sharded_param_grad is not None
            self._sharded_param_grad = DTensor(
                sharded_param_grad,
                self._sharding_spec,
                requires_grad=False,
            )
        del self.param_data
        gc.collect()

        self._setattr_on_modules(self.sharded_param)
        self.sharded_state = ShardedState.SHARDED

    def set_sharded_param_grad(self):
        self.sharded_param.grad = self._sharded_param_grad

    def init_dtype_attrs(self, mp_policy: MixedPrecisionPolicy):
        param_dtype, reduce_dtype = mp_policy.param_dtype, mp_policy.reduce_dtype
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

    @torch.no_grad()
    def init_unsharded_param(
        self, padded_unsharded_data: torch.Tensor, padded_unsharded_grad: Optional[torch.Tensor], offset: int
    ):
        unsharded_param = padded_unsharded_data.narrow(0, offset, self._orig_numel).view(self._orig_size)
        if self.is_dtensor:
            unsharded_param = DTensor(unsharded_param, self._tp_spec, requires_grad=False)
        self._unsharded_param = nn.Parameter(
            unsharded_param,
            requires_grad=self.sharded_param.requires_grad,
        )
        if self.sharded_param.requires_grad:
            assert padded_unsharded_grad is not None
            unsharded_param_grad = padded_unsharded_grad.narrow(0, offset, self._orig_numel).view(self._orig_size)
            if self.is_dtensor:
                unsharded_param_grad = DTensor(unsharded_param_grad, self._tp_spec, requires_grad=False)
            self._unsharded_param_grad = unsharded_param_grad

    def to_sharded(self) -> None:
        self._setattr_on_modules(self.sharded_param)
        self.sharded_state = ShardedState.SHARDED

    # def to_sharded_post_forward(self) -> None:
    #     self._setattr_on_modules(self._sharded_post_forward_param)
    #     self.free_unsharded_param()
    #     self.sharded_state = ShardedState.SHARDED_POST_FORWARD

    def to_unsharded(self, training_state: Optional[TrainingState] = None) -> None:
        # Assume that the data has been allocated and all-gathered
        if training_state == TrainingState.FORWARD and torch.is_grad_enabled() and self.sharded_param.requires_grad:
            self._setattr_on_modules(self._unsharded_params_for_backward[-1])
        elif training_state == TrainingState.PRE_BACKWARD and self.sharded_param.requires_grad:
            self._setattr_on_modules(self._unsharded_params_for_backward.pop())
        else:
            self._setattr_on_modules(self._unsharded_param)
        # if self.sharded_state == ShardedState.SHARDED_POST_FORWARD:
        #     # The data is allocated in the default stream via the post-forward
        #     # reshard and must be kept alive for the next all-gather copy-in.
        #     # Since we call this method after the copy-out, the data's lifetime
        #     # is ensured without further synchronization.
        #     self._sharded_post_forward_param = None
        #     self._sharded_post_forward_param_data = None  # free
        self.sharded_state = ShardedState.UNSHARDED

    def _setattr_on_modules(self, param: nn.Parameter) -> None:
        unsafe_setattr_param(self._module_info.module, self._module_info.param_name, param)
        for shared_module, shared_param_name in zip(
            self._module_info.shared_modules, self._module_info.shared_param_names
        ):
            unsafe_setattr_param(shared_module, shared_param_name, param)


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
