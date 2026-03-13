from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn
from torch.distributed._functional_collectives import AsyncCollectiveTensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor.placement_types import Shard

from ._api import MixedPrecisionPolicy
from ._common import FSDPMeshInfo, _raise_assert_with_print
from ._tensor import RaggedShard, RaggedShardDTensor

if TYPE_CHECKING:
    from torch.distributed.tensor.placement_types import Placement


class ShardedState(Enum):
    SHARDED = auto()
    UNSHARDED = auto()


@dataclass
class ParamModuleInfo:
    # Parameter names are unprefixed, e.g. "weight", not "lin.weight"
    module: nn.Module
    param_name: str
    shared_modules: list[nn.Module] = field(default_factory=list)
    shared_param_names: list[str] = field(default_factory=list)


@dataclass
class ExtensionsData:
    # User-defined metadata passed from pre to post-all-gather
    all_gather_metadata: Any | None = None

    def clear(self) -> None:
        self.all_gather_metadata = None


class YaFSDPParam:
    orig_dtype: torch.dtype
    param_dtype: torch.dtype | None
    reduce_dtype: torch.dtype | None
    _orig_size: torch.Size
    sharded_numel: int
    sharded_param: nn.Parameter
    _all_gather_input: torch.Tensor
    reduce_scatter_output: torch.Tensor | None
    sharded_param_grad: torch.Tensor | None
    unsharded_param: nn.Parameter
    _unsharded_accumulated_grad: torch.Tensor | None
    unsharded_accumulated_grad: torch.Tensor | None

    def __init__(
        self,
        param: nn.Parameter,
        module_info: ParamModuleInfo,
        mesh_info: FSDPMeshInfo,
        post_forward_mesh_info: FSDPMeshInfo | None,
        device: torch.device,
    ):
        self._module_info: ParamModuleInfo = module_info
        self.mesh_info = mesh_info
        self.post_forward_mesh_info = post_forward_mesh_info
        self.device = device
        if param.device != device:
            raise AssertionError(
                f"Expects the parameter to already be moved to device {device} but got {param.device}"
            )
        if not param.is_contiguous():
            raise NotImplementedError(
                f"YaFSDP does not support non-contiguous parameters yet: {param.shape=} {param.stride()=}"
            )
        self.param = param
        self.param_data = param._local_tensor if isinstance(param, DTensor) else param
        self.unsharded_accumulated_grad = None
        self._param_fqn: str | None = None  # prefixed from root module

    @torch.no_grad()
    def init_sharded_param(
        self,
        sharded_param_data: torch.Tensor,
        sharded_param_grad: torch.Tensor | None,
        global_offset: int,
        shard_numels: tuple[int, ...],
    ) -> None:
        param = self.param
        param_data = self.param_data
        del self.param
        del self.param_data
        self.is_dtensor = isinstance(param, DTensor)
        fsdp_placement = RaggedShard(
            local_numel=sharded_param_data.numel(),
            global_offset=global_offset,
            shard_numels=shard_numels,
        )
        if self.is_dtensor:
            self._tp_spec = cast("DTensor", param)._spec
            dp_mesh, tp_mesh = self.mesh_info.mesh, self._tp_spec.mesh
            if dp_mesh is None or tp_mesh is None:
                raise AssertionError(
                    "YaFSDP requires the DP and model parallel TP/EP mesh to be not None but got: \n"
                    f"DP's mesh: {dp_mesh}\nTP/EP's mesh: {tp_mesh}"
                )
            self._spmd_mesh = DeviceMesh._concatenate([dp_mesh, tp_mesh])
            if len(self._tp_spec.placements) > 1 or self._tp_spec.placements[
                0
            ] != Shard(0):
                raise NotImplementedError(
                    f"YaFSDP only supports 1D EP, not {self._tp_spec.placements}."
                )
            if self._spmd_mesh.ndim != 2:  # noqa: PLR2004
                raise AssertionError(
                    "_spmd_mesh.ndim can only be 2 (FSDP+EP) "
                    f"but got {self._spmd_mesh.ndim}."
                )
            self._spmd_placements: tuple[Placement, ...]
            dp_shard_tp_placement = (
                fsdp_placement,
                *self._tp_spec.placements,
            )
            self._spmd_placements = dp_shard_tp_placement
            self._sharding_spec = DTensorSpec(
                self._spmd_mesh,
                self._spmd_placements,
                tensor_meta=self._tp_spec.tensor_meta,
            )
        else:
            self._spmd_mesh = self.mesh_info.mesh
            self._spmd_placements = (fsdp_placement,)
            self._sharding_spec = DTensorSpec(
                self._spmd_mesh,
                self._spmd_placements,
                tensor_meta=TensorMeta(param.size(), param.stride(), param.dtype),
            )
        if not param_data.is_contiguous():
            raise AssertionError(
                f"Expected contiguous tensor, got {param_data.shape=} {param_data.stride()=}"
            )
        self._orig_size = param_data.size()

        self.sharded_numel = sharded_param_data.numel()
        self._sharded_param_data = param_data.new_zeros(self.sharded_numel)
        if hasattr(self._sharded_param_data, "__tensor_flatten__"):
            inner_tensors = [
                getattr(self._sharded_param_data, attr_name)
                for attr_name in self._sharded_param_data.__tensor_flatten__()[0]
            ]
        else:
            inner_tensors = [self._sharded_param_data]
        torch.utils.swap_tensors(inner_tensors[0], sharded_param_data)
        self.sharded_param = nn.Parameter(
            self.to_sharded_dtensor(self._sharded_param_data),
            requires_grad=param.requires_grad,
        )
        if self.sharded_param.requires_grad:
            assert sharded_param_grad is not None
            self.sharded_param_grad = self.to_sharded_dtensor(sharded_param_grad)
        else:
            self.sharded_param_grad = None
        self._setattr_on_modules(self.sharded_param)
        self.sharded_state = ShardedState.SHARDED
        self._init_extensions()

    def init_dtype_attrs(self, mp_policy: MixedPrecisionPolicy) -> None:
        param_dtype, reduce_dtype = (mp_policy.param_dtype, mp_policy.reduce_dtype)
        self.orig_dtype = self.param.dtype
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

    def init_all_gather_input(self, all_gather_input: torch.Tensor) -> None:
        self._all_gather_input = all_gather_input

    def init_reduce_scatter_output(self, reduce_scatter_output: torch.Tensor) -> None:
        self.reduce_scatter_output = reduce_scatter_output

    def init_unsharded_param(
        self,
        all_gather_output: torch.Tensor,
        reduce_scatter_input: torch.Tensor | None,
    ) -> None:
        inner_tensor = self._sharded_local_tensor
        unsharded_tensor = all_gather_output.view(self._orig_size)
        if hasattr(inner_tensor, "fsdp_post_all_gather"):
            unsharded_tensor, _ = inner_tensor.fsdp_post_all_gather(
                (unsharded_tensor,),
                self._extensions_data.all_gather_metadata,
                self.param_dtype or self.orig_dtype,
            )
        if self.is_dtensor:
            unsharded_tensor = DTensor(
                unsharded_tensor, self._tp_spec, requires_grad=False
            )
        self.unsharded_param = nn.Parameter(
            unsharded_tensor,
            requires_grad=self.sharded_param.requires_grad,
        )
        if self.unsharded_param.requires_grad:
            assert reduce_scatter_input is not None
            unsharded_accumulated_grad = reduce_scatter_input.view(self._orig_size)
            self._unsharded_accumulated_grad = unsharded_accumulated_grad
        else:
            self._unsharded_accumulated_grad = None

    def to_sharded(self) -> None:
        self._setattr_on_modules(self.sharded_param)
        self.sharded_state = ShardedState.SHARDED

    def to_unsharded(self) -> None:
        # Assume that the data has been allocated and all-gathered
        self._setattr_on_modules(self.unsharded_param)
        self.sharded_state = ShardedState.UNSHARDED

    def _setattr_on_modules(self, param: nn.Parameter) -> None:
        unsafe_setattr_param(
            self._module_info.module, self._module_info.param_name, param
        )
        for shared_module, shared_param_name in zip(
            self._module_info.shared_modules,
            self._module_info.shared_param_names,
            strict=True,
        ):
            unsafe_setattr_param(shared_module, shared_param_name, param)

    def to_sharded_dtensor(self, tensor: torch.Tensor) -> RaggedShardDTensor:
        if tensor.shape != (sharded_size := torch.Size((self.sharded_numel,))):
            _raise_assert_with_print(
                f"Expects size {sharded_size} but got {tensor.shape}"
            )
        return RaggedShardDTensor(
            tensor,
            self._sharding_spec,
            requires_grad=tensor.requires_grad,
        )

    def set_all_gather_input(self) -> None:
        self._assert_in_states(ShardedState.SHARDED)
        if hasattr(self._sharded_local_tensor, "fsdp_pre_all_gather"):
            sharded_local_tensor = self._sharded_local_tensor
            (
                all_gather_inputs,
                self._extensions_data.all_gather_metadata,
            ) = sharded_local_tensor.fsdp_pre_all_gather(self.shard_mesh_from_root)
            (all_gather_input,) = all_gather_inputs
            assert self._all_gather_input is not None
            self._all_gather_input.copy_(all_gather_input)

    @property
    def unsharded_grad_data(self) -> torch.Tensor:
        grad = self.unsharded_param.grad
        if grad is None:
            raise AssertionError("Expects unsharded_param.grad to not be None")
        return self._get_grad_inner_tensor(grad)

    def _get_grad_inner_tensor(self, grad: torch.Tensor) -> torch.Tensor:
        if self.is_dtensor:
            if isinstance(grad, AsyncCollectiveTensor):
                grad = grad.wait()
            if not isinstance(grad, DTensor):
                raise AssertionError(f"Expected DTensor, got {type(grad)}")
            placements = self._tp_spec.placements
            if placements != grad.placements:
                if len(self._tp_spec.placements) != len(grad.placements):
                    raise AssertionError(
                        f"Expected same placement length: {self._tp_spec=} {grad.placements=}"
                    )
                grad = grad.redistribute(placements=placements)
            grad = grad._local_tensor
        return grad

    @property
    def _sharded_local_tensor(self) -> torch.Tensor:
        return cast("DTensor", self.sharded_param)._local_tensor

    @property
    def shard_mesh(self) -> DeviceMesh:
        mesh = self.mesh_info.mesh
        if mesh.ndim == 1:
            return mesh
        elif mesh.ndim == 2:  # noqa: PLR2004
            if mesh.mesh_dim_names is None:
                raise AssertionError("Expected mesh_dim_names to not be None")
            return mesh[mesh.mesh_dim_names[-1]]
        raise ValueError(f"Invalid mesh: {mesh}")

    @property
    def shard_mesh_from_root(self) -> DeviceMesh:
        mesh = self.mesh_info.mesh

        if mesh.ndim == 1:
            return mesh
        else:
            if mesh.mesh_dim_names is None:
                raise AssertionError("Expected mesh_dim_names to not be None")
            shard_dim_name = mesh.mesh_dim_names[-1]
            return mesh[shard_dim_name]

    def _assert_in_states(self, *states: ShardedState) -> None:
        if self.sharded_state not in states:
            _raise_assert_with_print(
                f"Expects to be in one of {states}, not {self.sharded_state}"
            )

    def reset_sharded_param(self) -> None:
        module_info = self._module_info
        new_param = getattr(module_info.module, module_info.param_name)
        self._sharded_param_data.copy_(new_param._local_tensor)
        self.sharded_param = nn.Parameter(
            self.to_sharded_dtensor(self._sharded_param_data),
            requires_grad=self.sharded_param.requires_grad,
        )
        self._setattr_on_modules(self.sharded_param)

    def __repr__(self) -> str:
        return f"YaFSDPParam(fqn={self._param_fqn}, orig_size={self._orig_size})"


# NOTE: These bypass `nn.Module.__setattr__` checks, which incur non-trivial
# CPU overhead, if the module did not override it. For FSDP, we know we do not
# need those checks when transitioning between sharded/unsharded parameters.
def unsafe_setattr_param(
    module: nn.Module, param_name: str, param: nn.Parameter
) -> None:
    if getattr(module.__setattr__, "__func__", None) is nn.Module.__setattr__:
        module._parameters[param_name] = param
    else:  # slow path
        setattr(module, param_name, param)
