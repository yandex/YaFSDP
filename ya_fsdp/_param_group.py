import contextlib
import logging
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import _get_device_handle
from torch.distributed.fsdp._common_utils import _named_parameters_with_duplicates
from torch.profiler import record_function
from torch.utils._pytree import tree_flatten, tree_unflatten
from torch.utils.hooks import RemovableHandle

from ._api import FullStateDictConfig, MixedPrecisionPolicy, ShardedStateDictConfig, StateDictConfig, StateDictType
from ._collectives import all_gather, reduce_scatter
from ._common import FSDPMeshInfo, TrainingState
from ._param import ParamModuleInfo, ShardedState, YaFSDPParam

if TYPE_CHECKING:
    try:
        import yccl
    except ImportError:
        pass

logger = logging.getLogger("ya_fsdp")

_ModuleToHandleDict = Dict[nn.Module, RemovableHandle]  # for state dict


class YaFSDPBufferContext:
    """This has the buffer state for all-gather / reduce-scatter ops shared across YaFSDP parameter groups."""

    class BufferType(Enum):
        ALL_GATHER = auto()
        REDUCE_SCATTER = auto()

    def __init__(self, buffer_type: Optional[BufferType] = None, **kwargs):
        self._buffer_type = buffer_type

    def lazy_init(
        self,
        buffer_size: int,
        dtype: torch.dtype,
        device: torch.device,
        yccl_handle: Optional["yccl.Handle"] = None,
    ):
        if yccl_handle is not None and self._buffer_type == self.BufferType.REDUCE_SCATTER and dtype != torch.bfloat16:
            raise RuntimeError("YCCL requires reduce_dtype to be bfloat16")
        self.buffer = (
            torch.empty(buffer_size, dtype=dtype, device=device)
            if yccl_handle is None
            else getattr(
                yccl_handle,
                {
                    self.BufferType.ALL_GATHER: "add_all_gather_output_buffer",
                    self.BufferType.REDUCE_SCATTER: "add_reduce_scatter_buffer",
                }[self._buffer_type],
            )(buffer_size * torch.finfo(dtype).bits // torch.finfo(torch.bfloat16).bits).view(dtype)
        )
        self.owner: Optional[YaFSDPParamGroup] = None
        self.release_event: Optional[torch.Event] = None
        self.yccl_handle: Optional["yccl.Handle"] = yccl_handle


class YaFSDPCommContext:
    """This has the communication state shared across YaFSDP states/parameter groups."""

    def lazy_init(self, device: torch.device):
        self.device_handle = _get_device_handle(device.type)
        # Setting the all-gather/reduce-scatter streams to be higher priority
        # can help avoid some issues where their copies in/out are delayed and
        # block computation (this is different from high-pri NCCL streams)
        high_priority = -1
        # All-gather stream allows overlapping next all-gather with current
        # forward compute
        self.all_gather_stream = self.device_handle.Stream(priority=high_priority)
        # Reduce-scatter stream gives separate execution "thread" for post-
        # backward logic like pre/post-gradient division and reduce-scatter
        self.reduce_scatter_stream = self.all_gather_stream
        # Post-forward order for explicit backward prefetching
        self.post_forward_order: List[YaFSDPParamGroup] = []  # will cause ref cycles

    def get_all_gather_stream(self, training_state: TrainingState) -> torch.Stream:
        if training_state in (TrainingState.FORWARD, TrainingState.PRE_BACKWARD):
            return self.all_gather_stream
        current_stream = self.device_handle.current_stream()
        return current_stream


class YaFSDPParamGroup:
    """This class represents a parameter group to communicate together."""

    _orig_dtype: torch.dtype
    _param_dtype: Optional[torch.dtype]
    _reduce_dtype: Optional[torch.dtype]
    _all_gather_dtype: Optional[torch.dtype]
    _padded_unsharded_param_size: int
    _unsharded_param_numels: List[int]
    _padded_sharded_param_data: torch.Tensor
    _all_gather_input: torch.Tensor
    _padded_sharded_param_grad: Optional[torch.Tensor]
    _all_gather_output: torch.Tensor
    _padded_unsharded_param_grad: Optional[torch.Tensor]
    _padded_unsharded_param_grad_reduce_dtype: Optional[torch.Tensor]

    def __init__(
        self,
        params: List[nn.Parameter],
        modules: Tuple[nn.Module, ...],
        mesh_info: FSDPMeshInfo,
        post_forward_mesh_info: Optional[FSDPMeshInfo],
        device: torch.device,
        mp_policy: MixedPrecisionPolicy,
        shard_alignment: int,
    ):
        self.modules = modules  # permit ref cycle because 1:1 lifetime
        param_module_infos = _get_param_module_infos(params, modules)

        self.fsdp_params = [
            YaFSDPParam(
                param,
                module_info,
                mesh_info,
                post_forward_mesh_info,
                device,
            )
            for param, module_info in zip(params, param_module_infos)
        ]
        self.mesh_info = mesh_info
        self.post_forward_mesh_info = post_forward_mesh_info
        self.device = device
        self.device_handle = _get_device_handle(device.type)
        self.mp_policy = mp_policy
        self._training_state = TrainingState.IDLE
        # Group's sharded state always matches its parameters' sharded states
        self._sharded_state = ShardedState.SHARDED
        self._module_fqn: Optional[str] = None  # prefixed from root module

        # - Hook state
        self._module_to_pre_save_state_dict_hook_handle: _ModuleToHandleDict = {}
        self._module_to_pre_load_state_dict_hook_handle: _ModuleToHandleDict = {}

        # - Communication and communication/computation overlap
        self.comm_ctx = YaFSDPCommContext()
        # Group's indices in the shared post-forward order
        self._post_forward_indices: List[int] = []
        # Whether to reduce gradients at all (whether for FSDP or HSDP)
        self.reduce_grads: bool = True
        # Whether to reshard parameters after backward (only useful for
        # gradient accumulation)
        self.reshard_after_backward: bool = True
        # Whether to unshard in backward: can be overridden by the user if the
        # parameters in this group are not needed for backward (e.g. embedding)
        self.unshard_in_backward: bool = True

        # - CUDA events for stream synchronization
        self._all_gather_event: Optional[torch.Event] = None
        self._post_reduce_event: Optional[torch.Event] = None
        # # Holds the reshard-after-forward CUDA event when resharding to a
        # # different world size, which should be waited on in the next unshard
        # self._reshard_after_forward_event: Optional[torch.Event] = None

        self._init_mp_dtypes()
        param_group_requires_grad = any(param.requires_grad for param in params)

        self._data_buffer_ctx = YaFSDPBufferContext(buffer_type=YaFSDPBufferContext.BufferType.ALL_GATHER)
        self._grad_buffer_ctx = (
            YaFSDPBufferContext(buffer_type=YaFSDPBufferContext.BufferType.REDUCE_SCATTER)
            if param_group_requires_grad
            else None
        )
        self._reduce_dtype_grad_buffer_ctx = (
            YaFSDPBufferContext(buffer_type=YaFSDPBufferContext.BufferType.REDUCE_SCATTER)
            if param_group_requires_grad and self._reduce_dtype is not None
            else None
        )

        self._state_dict_type: StateDictType = StateDictType.SHARDED_STATE_DICT
        self._state_dict_config: StateDictConfig = ShardedStateDictConfig()

        shard_world_size = self.mesh_info.shard_mesh_size

        self._unsharded_param_numels = [fsdp_param.param_data.numel() for fsdp_param in self.fsdp_params]

        padded_unsharded_param_size = sum(self._unsharded_param_numels)
        divider = shard_world_size * shard_alignment
        if padded_unsharded_param_size % divider != 0:
            padded_unsharded_param_size += divider - padded_unsharded_param_size % divider
        self._padded_unsharded_param_size = padded_unsharded_param_size

        self._padded_sharded_param_data = torch.empty(
            padded_unsharded_param_size // shard_world_size, dtype=self._orig_dtype, device=device
        )

        self._all_gather_input = (
            self._padded_sharded_param_data
            if self._param_dtype is None and self._all_gather_dtype is None
            else torch.empty_like(self._padded_sharded_param_data, dtype=self._all_gather_dtype or self._param_dtype)
        )
        self._is_all_gather_input_set = False
        self._padded_sharded_param_grad = (
            torch.zeros_like(self._padded_sharded_param_data) if param_group_requires_grad else None
        )

        padded_unsharded_param_data = torch.empty(padded_unsharded_param_size, dtype=self._orig_dtype, device=device)
        assert len(self.fsdp_params) < (
            max_param_indices_dtype_value := torch.iinfo((param_indices_dtype := torch.uint16)).max
        )
        assert max(self._unsharded_param_numels) < (
            max_element_indices_dtype_value := torch.iinfo((element_indices_dtype := torch.int64)).max
        )
        padded_unsharded_param_indices = torch.full_like(
            padded_unsharded_param_data, fill_value=max_param_indices_dtype_value, dtype=param_indices_dtype
        )
        padded_unsharded_param_element_indices = torch.full_like(
            padded_unsharded_param_data, fill_value=max_element_indices_dtype_value, dtype=element_indices_dtype
        )
        for param_index, (
            fsdp_param,
            unsharded_param_numel,
            unsharded_param_data,
            unsharded_param_indices,
            unsharded_param_element_indices,
        ) in enumerate(
            zip(
                self.fsdp_params,
                self._unsharded_param_numels,
                padded_unsharded_param_data[: sum(self._unsharded_param_numels)].split(self._unsharded_param_numels),
                padded_unsharded_param_indices[: sum(self._unsharded_param_numels)].split(self._unsharded_param_numels),
                padded_unsharded_param_element_indices[: sum(self._unsharded_param_numels)].split(
                    self._unsharded_param_numels
                ),
            )
        ):
            unsharded_param_data.copy_(fsdp_param.param_data.view(-1))
            unsharded_param_indices.copy_(param_index)
            unsharded_param_element_indices.copy_(
                torch.arange(unsharded_param_numel, dtype=element_indices_dtype, device=device)
            )

        shard_rank = self.mesh_info.shard_mesh_rank

        self._padded_sharded_param_data.copy_(torch.chunk(padded_unsharded_param_data, shard_world_size)[shard_rank])

        padded_sharded_param_indices = torch.chunk(padded_unsharded_param_indices, shard_world_size)[shard_rank]
        padded_sharded_param_element_indices = torch.chunk(padded_unsharded_param_element_indices, shard_world_size)[
            shard_rank
        ]

        self._sharded_param_numels = [
            cast(int, padded_sharded_param_indices.eq(index).sum().item()) for index, _ in enumerate(self.fsdp_params)
        ]
        first_nonzero_local_numel_index = next(
            (index for index, numel in enumerate(self._sharded_param_numels) if numel > 0), -1
        )
        sharded_data_global_offsets = [
            (
                cast(int, element_indices.min().item())
                if (
                    element_indices := padded_sharded_param_element_indices[padded_sharded_param_indices.eq(index)]
                ).numel()
                > 0
                else (0 if index < first_nonzero_local_numel_index else self._unsharded_param_numels[index])
            )
            for index in range(len(self.fsdp_params))
        ]

        for fsdp_param, sharded_param_data, sharded_param_grad, global_offset in zip(
            self.fsdp_params,
            self._padded_sharded_param_data[: sum(self._sharded_param_numels)].split(self._sharded_param_numels),
            (
                self._padded_sharded_param_grad[: sum(self._sharded_param_numels)].split(self._sharded_param_numels)
                if self._padded_sharded_param_grad is not None
                else (None,) * len(self._sharded_param_numels)
            ),
            sharded_data_global_offsets,
        ):
            fsdp_param._init_sharded_param(sharded_param_data, sharded_param_grad, global_offset=global_offset)

    # Initialization #
    def _init_unsharded_params(self):
        self._all_gather_output = self._data_buffer_ctx.buffer.narrow(0, 0, self._padded_unsharded_param_size)
        self._padded_unsharded_param_grad = (
            self._grad_buffer_ctx.buffer.narrow(0, 0, self._padded_unsharded_param_size)
            if self._grad_buffer_ctx is not None
            else None
        )
        self._padded_unsharded_param_grad_reduce_dtype = (
            self._reduce_dtype_grad_buffer_ctx.buffer.narrow(0, 0, self._padded_unsharded_param_size)
            if self._reduce_dtype_grad_buffer_ctx is not None
            else None
        )

        for fsdp_param, all_gather_output, unsharded_param_grad in zip(
            self.fsdp_params,
            self._all_gather_output[: sum(self._unsharded_param_numels)].split(self._unsharded_param_numels),
            self._padded_unsharded_param_grad[: sum(self._unsharded_param_numels)].split(self._unsharded_param_numels)
            if self._padded_unsharded_param_grad is not None
            else (None,) * len(self._unsharded_param_numels),
        ):
            fsdp_param.init_all_gather_output(all_gather_output, unsharded_param_grad)

    def _init_mp_dtypes(self) -> None:
        for fsdp_param in self.fsdp_params:
            fsdp_param.init_dtype_attrs(self.mp_policy)
        orig_dtypes = {fsdp_param.orig_dtype for fsdp_param in self.fsdp_params}
        if len(orig_dtypes) != 1:
            # This can be relaxed if we copy-out for the reduce-scatter
            raise AssertionError(f"YaFSDP expects uniform original parameter dtype but got {orig_dtypes}")
        self._orig_dtype = next(iter(orig_dtypes))
        param_dtypes = {fsdp_param.param_dtype for fsdp_param in self.fsdp_params}
        if len(param_dtypes) != 1:
            raise AssertionError(f"YaFSDP expects uniform param dtype but got {param_dtypes}")
        self._param_dtype = next(iter(param_dtypes))
        reduce_dtypes = {fsdp_param.reduce_dtype for fsdp_param in self.fsdp_params}
        if len(reduce_dtypes) != 1:
            raise AssertionError(f"YaFSDP expects uniform reduce dtype but got {reduce_dtypes}")
        self._reduce_dtype = next(iter(reduce_dtypes))
        all_gather_dtypes = {
            fsdp_param.param_data._dtype if hasattr(fsdp_param.param_data, "fsdp_pre_all_gather") else None
            for fsdp_param in self.fsdp_params
        }
        if len(all_gather_dtypes) != 1:
            raise AssertionError(f"YaFSDP expects uniform unsharded data dtype but got {all_gather_dtypes}")
        self._all_gather_dtype = next(iter(all_gather_dtypes))

    def lazy_init(self):
        # Lazy init should be idempotent
        if not hasattr(self.comm_ctx, "device_handle"):
            self.comm_ctx.device_handle = _get_device_handle(self.device.type)
        self._validate_no_meta_params()
        self._register_state_dict_hooks()
        self._init_unsharded_params()
        if (yccl_handle := self._data_buffer_ctx.yccl_handle) is not None:
            self._all_gather_input = (
                yccl_handle.add_all_gather_input_buffer(self._all_gather_input.view(torch.bfloat16).numel())
                .view(self._all_gather_input.dtype)
                .copy_(self._all_gather_input)
            )
        for fsdp_param, all_gather_input in zip(
            self.fsdp_params,
            self._all_gather_input[: sum(self._sharded_param_numels)].split(self._sharded_param_numels),
        ):
            fsdp_param.init_all_gather_input(all_gather_input)
        if (grad_buffer_ctx := self._grad_buffer_ctx) is not None and (
            yccl_handle := grad_buffer_ctx.yccl_handle
        ) is not None:
            if self._reduce_dtype is not None:
                raise NotImplementedError("YCCL requires param_dtype and reduce_dtype to be the same.")

    # Runtime #
    def unshard(self):
        if self._all_gather_event is not None:  # already called, pending wait
            return
        if self.is_unsharded:
            return  # no-op
        if not self.unshard_in_backward and self._training_state == TrainingState.PRE_BACKWARD:
            return
        logger.debug("%s", self._with_fqn(f"YaFSDP::{self._training_state.name.lower()}_unshard"))
        with record_function(self._with_fqn("YaFSDP::all_gather")):
            self._all_gather_event = all_gather(
                self,
                self._padded_sharded_param_data,
                self._all_gather_input,
                self._all_gather_output,
                self._data_buffer_ctx,
                self._all_gather_process_group,
                self.comm_ctx.get_all_gather_stream(self._training_state),
                self._param_dtype,
                self._all_gather_dtype,
                self.device_handle,
                self._data_buffer_ctx.yccl_handle,
            )

    def wait_for_unshard(
        self,
        register_post_backward_hook: bool = False,
        args: Optional[Tuple[Any, ...]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        if self._all_gather_event:
            self.device_handle.current_stream().wait_event(self._all_gather_event)
        self._all_gather_event = None
        for fsdp_param in self.fsdp_params:
            fsdp_param.init_unsharded_param()
        if register_post_backward_hook:
            args, kwargs, unsharded_params = self._register_post_backward_hook(args, kwargs)
            for fsdp_param, unsharded_param in unsharded_params.items():
                fsdp_param.register_unsharded_param_with_post_backward_hook(unsharded_param)
        self._to_unsharded()
        if register_post_backward_hook:
            return args, kwargs

    def reshard(self):
        if self._training_state == TrainingState.FORWARD:
            if not self._reshard_after_forward:
                return
        logger.debug("%s", self._with_fqn(f"YaFSDP::{self._training_state.name.lower()}_reshard"))
        self._to_sharded()
        self._data_buffer_ctx.release_event = self.device_handle.current_stream().record_event()
        self._data_buffer_ctx.owner = None

    def pre_forward(
        self, module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        logger.debug("%s", self._with_fqn("YaFSDP::pre_forward"))
        with record_function(self._with_fqn("YaFSDP::pre_forward")):
            self._training_state = TrainingState.FORWARD
            self.unshard()
            args, kwargs = self.wait_for_unshard(register_post_backward_hook=True, args=args, kwargs=kwargs)
            return args, kwargs

    def post_forward(self, module: nn.Module, input: Any, output: Any):
        logger.debug("%s", self._with_fqn("YaFSDP::post_forward"))
        with record_function(self._with_fqn("YaFSDP::post_forward")):
            self.reshard()
            if torch.is_grad_enabled():
                self._record_post_forward()
            self._training_state = TrainingState.IDLE
            return output

    def _record_post_forward(self) -> None:
        # Since a group has one pre-backward unshard for each forward call
        # before the backward, we record each usage (with multiplicity)
        post_forward_index = len(self.comm_ctx.post_forward_order)
        self.comm_ctx.post_forward_order.append(self)
        self._post_forward_indices.append(post_forward_index)

    def pre_backward(self, default_prefetch: bool, *unused: Any):
        if self._training_state == TrainingState.PRE_BACKWARD:
            return
        logger.debug("%s", self._with_fqn("YaFSDP::pre_backward"))
        with record_function(self._with_fqn("YaFSDP::pre_backward")):
            self._training_state = TrainingState.PRE_BACKWARD
            self.unshard()
            self.wait_for_unshard()
            if default_prefetch:
                self._backward_prefetch()
        if self._grad_buffer_ctx is None:
            return
        if (owner := self._grad_buffer_ctx.owner) is not None and owner != self:
            raise RuntimeError(f"{self} tried to acquire its gradient buffer, but it is in use by {owner}")
        if (release_event := self._grad_buffer_ctx.release_event) is not None:
            self.device_handle.current_stream().wait_event(release_event)
            self._grad_buffer_ctx.release_event = None
        self._grad_buffer_ctx.owner = self

    def post_backward(self, *unused: Any):
        # This method should be idempotent and safe to call even when this
        # FSDP parameter group was not used in backward (should be a no-op)
        logger.debug("%s", self._with_fqn("YaFSDP::post_backward"))
        self._training_state = TrainingState.POST_BACKWARD
        with record_function(self._with_fqn("YaFSDP::post_backward_reshard")):
            if not self.reduce_grads:
                if self.reshard_after_backward:
                    self.reshard()
                return
            fsdp_params_with_grad: List[YaFSDPParam] = []
            for fsdp_param in self.fsdp_params:
                if fsdp_param._unsharded_param.grad is not None:
                    fsdp_params_with_grad.append(fsdp_param)
                    fsdp_param._unsharded_param.grad = None
            if self.reshard_after_backward:
                self.reshard()
            # we prefetch here and not in pre_backward to avoid prefetching a layer into
            # the same buffer the layer we're performing backward on is using
        if self._grad_buffer_ctx is None:
            assert len(fsdp_params_with_grad) == 0
            return
        if len(fsdp_params_with_grad) != 0:
            logger.debug("%s", self._with_fqn("YaFSDP::post_backward_reduce"))
            with record_function(self._with_fqn("YaFSDP::post_backward_reduce")):
                self._post_reduce_event, grad_buffer_release_event = reduce_scatter(
                    self,
                    fsdp_params_with_grad,
                    cast(torch.Tensor, self._padded_sharded_param_grad),
                    cast(torch.Tensor, self._padded_unsharded_param_grad),
                    self._padded_unsharded_param_grad_reduce_dtype,
                    self._reduce_dtype_grad_buffer_ctx,
                    self._reduce_scatter_process_group,
                    self.comm_ctx.reduce_scatter_stream,
                    self._orig_dtype,
                    self._param_dtype,
                    self._reduce_dtype,
                    self.device_handle,
                    self.mp_policy.bit32_acc_for_bit16_reduce_scatter,
                    self._grad_buffer_ctx.yccl_handle,
                )
        else:
            grad_buffer_release_event = self.device_handle.current_stream().record_event()
        self._grad_buffer_ctx.release_event = grad_buffer_release_event
        self._grad_buffer_ctx.owner = None

    def is_sharded_param_grad_set(self) -> bool:
        grad_is_set = any(
            fsdp_param.sharded_param.grad is not None
            for fsdp_param in self.fsdp_params
            if fsdp_param.sharded_param.requires_grad
        )
        return grad_is_set

    def finalize_backward(self):
        self._wait_for_post_backward()
        if self._all_gather_event is not None:
            # If there was a mistargeted unshard without a corresponding wait,
            # then we wait here and clear the unshard
            logger.debug("%s", self._with_fqn("YaFSDP::wait_for_mistargeted_unshard"))
            self.device_handle.current_stream().wait_event(self._all_gather_event)
            self._all_gather_event = None
            self._data_buffer_ctx.owner = None
        self._post_forward_indices.clear()
        for fsdp_param in self.fsdp_params:
            self._is_all_gather_input_set = False
            if fsdp_param.sharded_param.requires_grad:
                assert len(fsdp_param._unsharded_params_with_post_backward_hook) == 0

    def _wait_for_post_backward(self):
        if self._post_reduce_event is not None:
            self.device_handle.current_stream().wait_event(self._post_reduce_event)
            self._post_reduce_event = None
            assert self._grad_buffer_ctx.owner is None
            self._grad_buffer_ctx.release_event = None
            if self._reduce_dtype is not None:
                assert self._reduce_dtype_grad_buffer_ctx.owner is None
                self._reduce_dtype_grad_buffer_ctx.release_event = None

    def _backward_prefetch(self) -> None:
        if not self._post_forward_indices:
            # Can be cleared if running multiple `backward`s
            return
        curr_index = self._post_forward_indices.pop()
        if (target_index := curr_index - 1) < 0:
            return
        # Prefetch naively using the reverse post-forward order, which may
        # have mistargeted prefetches if not all modules used in forward
        # are used in this backward
        target_fsdp_param_group = self.comm_ctx.post_forward_order[target_index]
        if target_fsdp_param_group._data_buffer_ctx is not self._data_buffer_ctx:
            self._prefetch_unshard(target_fsdp_param_group, "backward")

    @staticmethod
    def _prefetch_unshard(target_fsdp_param_group: "YaFSDPParamGroup", pass_type: str) -> None:
        if pass_type == "backward":
            training_state = TrainingState.PRE_BACKWARD
        elif pass_type == "forward":
            training_state = TrainingState.FORWARD
        else:
            raise ValueError(f"Unknown pass type: {pass_type}")
        target_fqn = target_fsdp_param_group._module_fqn
        logger.debug("%s", f"YaFSDP::{pass_type}_prefetch for {target_fqn}")
        with record_function(
            f"YaFSDP::{pass_type}_prefetch for {target_fqn}"
        ), target_fsdp_param_group.use_training_state(training_state):
            target_fsdp_param_group.unshard()

    # Utilities #
    def _to_sharded(self):
        if not self.is_sharded:
            for fsdp_param in self.fsdp_params:
                fsdp_param.to_sharded(self._training_state)
            self._sharded_state = ShardedState.SHARDED

    def _to_unsharded(self):
        for fsdp_param in self.fsdp_params:
            fsdp_param.to_unsharded(self._training_state)
        if not self.is_unsharded:
            self._sharded_state = ShardedState.UNSHARDED

    @property
    def is_sharded(self) -> bool:
        return self._sharded_state == ShardedState.SHARDED

    @property
    def is_unsharded(self) -> bool:
        return self._sharded_state == ShardedState.UNSHARDED

    @contextlib.contextmanager
    def use_training_state(self, training_state: TrainingState):
        old_training_state = self._training_state
        self._training_state = training_state
        try:
            yield
        finally:
            self._training_state = old_training_state

    # Hook Registration #
    def _register_post_backward_hook(
        self,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any], Dict[YaFSDPParam, nn.Parameter]]:
        if not torch.is_grad_enabled():
            return args, kwargs, {}
        args_list, args_spec = tree_flatten(args)
        kwargs_list, kwargs_spec = tree_flatten(kwargs)
        args_kwargs_list = list(args_list) + list(kwargs_list)
        inp_tensor_indices: List[int] = []
        inp_tensors: List[torch.Tensor] = []
        for i, obj in enumerate(args_kwargs_list):
            if torch.is_tensor(obj) and obj.requires_grad:
                inp_tensor_indices.append(i)
                inp_tensors.append(obj)
        fsdp_params_with_grads = [
            fsdp_param for fsdp_param in self.fsdp_params if fsdp_param.sharded_param.requires_grad
        ]
        inp_tensors = RegisterPostBackwardFunction.apply(
            self,
            *(fsdp_param._unsharded_param for fsdp_param in fsdp_params_with_grads),
            *inp_tensors,
        )
        unsharded_params, inp_tensors = (
            inp_tensors[: len(fsdp_params_with_grads)],
            inp_tensors[len(fsdp_params_with_grads) :],
        )
        for unsharded_param in unsharded_params:
            unsharded_param._is_param = True
        fsdp_param2unsharded_param = {
            fsdp_param: cast(nn.Parameter, unsharded_param)
            for fsdp_param, unsharded_param in zip(fsdp_params_with_grads, unsharded_params, strict=True)
        }
        if len(inp_tensors) == 0:
            return args, kwargs, fsdp_param2unsharded_param  # no tensors that require gradients
        for inp_tensor_idx, inp_tensor in zip(inp_tensor_indices, inp_tensors):
            args_kwargs_list[inp_tensor_idx] = inp_tensor
        args_list = args_kwargs_list[: len(args_list)]
        kwargs_list = args_kwargs_list[len(args_list) :]
        args = tree_unflatten(args_list, args_spec)
        kwargs = tree_unflatten(kwargs_list, kwargs_spec)
        return args, kwargs, fsdp_param2unsharded_param

    def _register_state_dict_hooks(self) -> None:
        num_pre_save_hooks = len(self._module_to_pre_save_state_dict_hook_handle)
        num_pre_load_hooks = len(self._module_to_pre_load_state_dict_hook_handle)
        assert (
            num_pre_save_hooks == num_pre_load_hooks
        ), f"Pre-save: {num_pre_save_hooks} pre-load: {num_pre_load_hooks}"
        if num_pre_save_hooks > 0:
            return  # already registered

        def to_sharded_hook(*args: Any, **kwargs: Any) -> None:
            self._to_sharded()

        def unshard_hook(*args: Any, **kwargs: Any) -> None:
            logger.debug(
                "%s",
                self._with_fqn(
                    f"YaFSDP::unshard_hook ({self.mesh_info.shard_mesh_rank}, {self.mesh_info.intra_node_group.rank()})"
                ),
            )
            self.unshard()
            self.wait_for_unshard()

        def state_dict_pre_hook(*args: Any, **kwargs: Any) -> None:
            _state_dict_pre_hook_fn = {
                StateDictType.FULL_STATE_DICT: unshard_hook,
                StateDictType.SHARDED_STATE_DICT: to_sharded_hook,
            }
            _state_dict_pre_hook_fn[self._state_dict_type](*args, **kwargs)

        def load_state_dict_pre_hook(
            module: nn.Module,
            state_dict: Dict[str, Any],
            prefix: str,
            local_metadata: Dict[str, Any],
            *args: Any,
        ) -> None:
            if self._state_dict_type == StateDictType.FULL_STATE_DICT:
                raise ValueError("Full state dict loading is not implemented.")
            if (version := local_metadata.get("version")) != 2:
                raise ValueError(f"Unsupported state dict version: {version}")
            self._to_sharded()

        def rank0_only_hook(module: nn.Module, state_dict: Dict[str, Any], *args: Any) -> None:
            logger.debug(
                "%s",
                self._with_fqn(
                    f"YaFSDP::rank0_only_hook ({self.mesh_info.shard_mesh_rank}, {self.mesh_info.intra_node_group.rank()})"
                ),
            )
            if self.mesh_info.intra_node_group.rank() != 0:
                state_dict.clear()

        def detach_and_clone_hook(module: nn.Module, state_dict: Dict[str, Any], *args: Any) -> None:
            logger.debug(
                "%s",
                self._with_fqn(
                    f"YaFSDP::detach_and_clone_hook ({self.mesh_info.shard_mesh_rank}, {self.mesh_info.intra_node_group.rank()})"
                ),
            )
            for k, v in state_dict.items():
                state_dict[k] = v.detach().clone()

        def offload_to_cpu_hook(module: nn.Module, state_dict: Dict[str, Any], *args: Any) -> None:
            logger.debug(
                "%s",
                self._with_fqn(
                    f"YaFSDP::offload_to_cpu_hook ({self.mesh_info.shard_mesh_rank}, {self.mesh_info.intra_node_group.rank()})"
                ),
            )
            for k, v in state_dict.items():
                state_dict[k] = v.cpu()

        def reshard_hook(*args: Any) -> None:
            logger.debug(
                "%s",
                self._with_fqn(
                    f"YaFSDP::reshard_hook ({self.mesh_info.shard_mesh_rank}, {self.mesh_info.intra_node_group.rank()})"
                ),
            )
            self.reshard()

        def state_dict_post_hook(module: nn.Module, state_dict: Dict[str, Any], *args: Any) -> None:
            if (
                self._state_dict_type is StateDictType.FULL_STATE_DICT
                and cast(FullStateDictConfig, self._state_dict_config).rank0_only
            ):
                rank0_only_hook(module, state_dict, *args)
            if self._state_dict_type is StateDictType.FULL_STATE_DICT:
                detach_and_clone_hook(module, state_dict, *args)
            if self._state_dict_config.offload_to_cpu:
                offload_to_cpu_hook(module, state_dict, *args)
            if self._state_dict_type is StateDictType.FULL_STATE_DICT:
                reshard_hook(module, state_dict, *args)

        for module in self.modules:
            self._module_to_pre_save_state_dict_hook_handle[module] = module.register_state_dict_pre_hook(
                state_dict_pre_hook
            )
            self._module_to_pre_load_state_dict_hook_handle[module] = module.register_load_state_dict_pre_hook(
                load_state_dict_pre_hook
            )
            module._register_state_dict_hook(state_dict_post_hook)

    # Properties #
    @property
    def _reshard_after_forward(self) -> bool:
        return self.post_forward_mesh_info is not None

    @property
    def _all_gather_process_group(self) -> dist.ProcessGroup:
        mesh_info = self.mesh_info
        assert isinstance(mesh_info, FSDPMeshInfo)
        return mesh_info.shard_process_group

    @property
    def _reduce_scatter_process_group(self) -> dist.ProcessGroup:
        assert isinstance(self.mesh_info, FSDPMeshInfo)
        return self.mesh_info.shard_process_group

    def _with_fqn(self, label: str) -> str:
        if self._module_fqn:
            return f"{label} ({self._module_fqn})"
        return label

    def __repr__(self):
        return f"YaFSDPParamGroup(fqn={self._module_fqn})"

    def _validate_no_meta_params(self):
        param_names_on_meta = [
            fsdp_param._param_fqn for fsdp_param in self.fsdp_params if fsdp_param.sharded_param.device.type == "meta"
        ]
        if param_names_on_meta:
            raise RuntimeError(
                "FSDP parameters should be materialized from meta device before training, "
                f"but the following were still on meta device: {param_names_on_meta}\n"
                "For example, call module.to_empty(device) to materialize to device and "
                "call module.reset_parameters() on each module to initialize values."
            )


def _get_param_module_infos(params: List[nn.Parameter], modules: Tuple[nn.Module, ...]) -> List[ParamModuleInfo]:
    """
    Shared parameter: lin1.weight = lin2.weight
    Shared module: mlp.lin1 = mlp.lin2
    We do not remove duplicates when traversing both modules and parameters to
    find shared modules' parameters and shared parameters within a module.
    """
    params_set = set(params)
    param_to_module_info: Dict[nn.Parameter, ParamModuleInfo] = {}
    for module in modules:
        for _, submodule in module.named_modules(remove_duplicate=False):
            for param_name, param in _named_parameters_with_duplicates(submodule, recurse=False):
                if param in params_set:
                    if param not in param_to_module_info:
                        param_to_module_info[param] = ParamModuleInfo(submodule, param_name)
                    else:
                        param_to_module_info[param].shared_modules.append(submodule)
                        param_to_module_info[param].shared_param_names.append(param_name)
    if len(param_to_module_info) != len(params):
        raise AssertionError(f"Some parameters are not in the module tree of {module}")
    return [param_to_module_info[param] for param in params]


class RegisterPostBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, param_group: YaFSDPParamGroup, *inputs: torch.Tensor):
        # All tensors in `inputs` should require gradient
        ctx.param_group = param_group
        ctx.set_materialize_grads(False)
        return inputs

    @staticmethod
    def backward(ctx, *grads: torch.Tensor):
        param_group: YaFSDPParamGroup = ctx.param_group
        for fsdp_param in param_group.fsdp_params:
            if fsdp_param.sharded_param.requires_grad:
                del fsdp_param._unsharded_params_with_post_backward_hook[-1]
        ctx.param_group.post_backward()
        return (None,) + grads


class MultiDtypeYaFSDPBufferContext(YaFSDPBufferContext):
    def __init__(self, buffer_type: YaFSDPBufferContext.BufferType, mp_policy: MixedPrecisionPolicy):
        self._all_gather_dtype_to_buffer_ctx: Dict[Optional[torch.dtype], YaFSDPBufferContext] = {
            all_gather_dtype: YaFSDPBufferContext(buffer_type)
            for all_gather_dtype in [mp_policy.param_dtype, *set(mp_policy.all_gather_dtype_to_param_cls)]
        }

    def lazy_init(
        self,
        buffer_size: Dict[torch.dtype, int],
        dtype: torch.dtype,
        device: torch.device,
        yccl_handle: Optional["yccl.Handle"] = None,
    ):
        for all_gather_dtype, size in buffer_size.items():
            (buffer_ctx := self._all_gather_dtype_to_buffer_ctx[all_gather_dtype]).lazy_init(
                size,
                all_gather_dtype if buffer_ctx._buffer_type == self.BufferType.ALL_GATHER else dtype,
                device,
                yccl_handle,
            )

    @property
    def yccl_handle(self) -> torch.dtype:
        return next(iter(buffer_ctx.yccl_handle for buffer_ctx in self._all_gather_dtype_to_buffer_ctx.values()))


class MultiDtypeYaFSDPParamGroup(YaFSDPParamGroup):
    def __init__(
        self,
        params: List[nn.Parameter],
        modules: Tuple[nn.Module, ...],
        mesh_info: FSDPMeshInfo,
        post_forward_mesh_info: Optional[FSDPMeshInfo],
        device: torch.device,
        mp_policy: MixedPrecisionPolicy,
        shard_alignment: int,
    ):
        all_gather_dtype_to_params = {
            **{
                mp_policy.param_dtype: [
                    param
                    for param in params
                    if not any(
                        isinstance(param, param_cls) for param_cls in mp_policy.all_gather_dtype_to_param_cls.values()
                    )
                ]
            },
            **{
                all_gather_dtype: [param for param in params if isinstance(param, param_cls)]
                for all_gather_dtype, param_cls in mp_policy.all_gather_dtype_to_param_cls.items()
            },
        }
        self._all_gather_dtype_to_param_group = {
            all_gather_dtype: YaFSDPParamGroup(
                dtype_params,
                tuple(
                    module
                    for module in modules
                    if any(param is dtype_param for param in module.parameters() for dtype_param in dtype_params)
                ),
                mesh_info,
                post_forward_mesh_info,
                device,
                mp_policy,
                shard_alignment,
            )
            for all_gather_dtype, dtype_params in all_gather_dtype_to_params.items()
            if dtype_params
        }
        self.__data_buffer_ctx = MultiDtypeYaFSDPBufferContext(
            buffer_type=MultiDtypeYaFSDPBufferContext.BufferType.ALL_GATHER,
            mp_policy=mp_policy,
        )
        self.__grad_buffer_ctx = (
            MultiDtypeYaFSDPBufferContext(
                buffer_type=MultiDtypeYaFSDPBufferContext.BufferType.REDUCE_SCATTER,
                mp_policy=mp_policy,
            )
            if any(
                param_group._grad_buffer_ctx is not None
                for param_group in self._all_gather_dtype_to_param_group.values()
            )
            else None
        )
        self.__reduce_dtype_grad_buffer_ctx = (
            MultiDtypeYaFSDPBufferContext(
                buffer_type=MultiDtypeYaFSDPBufferContext.BufferType.REDUCE_SCATTER,
                mp_policy=mp_policy,
            )
            if any(
                param_group._reduce_dtype_grad_buffer_ctx is not None
                for param_group in self._all_gather_dtype_to_param_group.values()
            )
            else None
        )

    @property
    def modules(self):
        return list(
            {module for param_group in self._all_gather_dtype_to_param_group.values() for module in param_group.modules}
        )

    @property
    def fsdp_params(self):
        return [
            fsdp_param
            for param_group in self._all_gather_dtype_to_param_group.values()
            for fsdp_param in param_group.fsdp_params
        ]

    @property
    def mesh_info(self) -> bool:
        return next(iter(param_group.mesh_info for param_group in self._all_gather_dtype_to_param_group.values()))

    @property
    def _training_state(self) -> bool:
        return next(iter(param_group._training_state for param_group in self._all_gather_dtype_to_param_group.values()))

    @_training_state.setter
    def _training_state(self, value):
        for param_group in self._all_gather_dtype_to_param_group.values():
            param_group._training_state = value

    @property
    def _module_fqn(self) -> bool:
        return next(iter(param_group._module_fqn for param_group in self._all_gather_dtype_to_param_group.values()))

    @_module_fqn.setter
    def _module_fqn(self, value):
        for param_group in self._all_gather_dtype_to_param_group.values():
            param_group._module_fqn = value

    @property
    def comm_ctx(self):
        return next(iter(param_group.comm_ctx for param_group in self._all_gather_dtype_to_param_group.values()))

    @comm_ctx.setter
    def comm_ctx(self, value):
        for param_group in self._all_gather_dtype_to_param_group.values():
            param_group.comm_ctx = value

    @property
    def reduce_grads(self):
        return next(iter(param_group.reduce_grads for param_group in self._all_gather_dtype_to_param_group.values()))

    @reduce_grads.setter
    def reduce_grads(self, value):
        for param_group in self._all_gather_dtype_to_param_group.values():
            param_group.reduce_grads = value

    @property
    def reshard_after_backward(self):
        return next(
            iter(param_group.reshard_after_backward for param_group in self._all_gather_dtype_to_param_group.values())
        )

    @reshard_after_backward.setter
    def reshard_after_backward(self, value):
        for param_group in self._all_gather_dtype_to_param_group.values():
            param_group.reshard_after_backward = value

    @property
    def unshard_in_backward(self) -> bool:
        return next(
            iter(param_group.unshard_in_backward for param_group in self._all_gather_dtype_to_param_group.values())
        )

    @property
    def _data_buffer_ctx(self):
        return self.__data_buffer_ctx

    @_data_buffer_ctx.setter
    def _data_buffer_ctx(self, value):
        self.__data_buffer_ctx = value
        for param_dtype, param_group in self._all_gather_dtype_to_param_group.items():
            param_group._data_buffer_ctx = self._data_buffer_ctx._all_gather_dtype_to_buffer_ctx[param_dtype]

    @property
    def _grad_buffer_ctx(self):
        return self.__grad_buffer_ctx

    @_grad_buffer_ctx.setter
    def _grad_buffer_ctx(self, value):
        self.__grad_buffer_ctx = value
        for param_dtype, param_group in self._all_gather_dtype_to_param_group.items():
            param_group._grad_buffer_ctx = self._grad_buffer_ctx._all_gather_dtype_to_buffer_ctx[param_dtype]

    @property
    def _reduce_dtype_grad_buffer_ctx(self):
        return self.__reduce_dtype_grad_buffer_ctx

    @_reduce_dtype_grad_buffer_ctx.setter
    def _reduce_dtype_grad_buffer_ctx(self, value):
        self.__reduce_dtype_grad_buffer_ctx = value
        for param_dtype, param_group in self._all_gather_dtype_to_param_group.items():
            param_group._reduce_dtype_grad_buffer_ctx = (
                self._reduce_dtype_grad_buffer_ctx._all_gather_dtype_to_buffer_ctx[param_dtype]
            )

    @property
    def _state_dict_type(self):
        return next(
            iter(param_group._state_dict_type for param_group in self._all_gather_dtype_to_param_group.values())
        )

    @_state_dict_type.setter
    def _state_dict_type(self, value):
        for param_group in self._all_gather_dtype_to_param_group.values():
            param_group._state_dict_type = value

    @property
    def _state_dict_config(self):
        return next(
            iter(param_group._state_dict_config for param_group in self._all_gather_dtype_to_param_group.values())
        )

    @_state_dict_config.setter
    def _state_dict_config(self, value):
        for param_group in self._all_gather_dtype_to_param_group.values():
            param_group._state_dict_config = value

    @property
    def _padded_unsharded_param_size(self):
        return {
            all_gather_dtype: param_group._padded_unsharded_param_size
            for all_gather_dtype, param_group in self._all_gather_dtype_to_param_group.items()
        }

    @property
    def _orig_dtype(self) -> bool:
        return next(iter(param_group._orig_dtype for param_group in self._all_gather_dtype_to_param_group.values()))

    @property
    def _param_dtype(self) -> bool:
        return next(iter(param_group._param_dtype for param_group in self._all_gather_dtype_to_param_group.values()))

    @property
    def _reduce_dtype(self) -> bool:
        return next(iter(param_group._reduce_dtype for param_group in self._all_gather_dtype_to_param_group.values()))

    def lazy_init(self):
        for param_group in self._all_gather_dtype_to_param_group.values():
            param_group.lazy_init()

    def unshard(self):
        for param_group in self._all_gather_dtype_to_param_group.values():
            param_group.unshard()

    def wait_for_unshard(self):
        for param_group in self._all_gather_dtype_to_param_group.values():
            param_group.wait_for_unshard()

    def reshard(self):
        for param_group in self._all_gather_dtype_to_param_group.values():
            param_group.reshard()

    def pre_forward(
        self, module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        for param_group in self._all_gather_dtype_to_param_group.values():
            args, kwargs = param_group.pre_forward(module, args, kwargs)
        return args, kwargs

    def post_forward(self, module: nn.Module, input: Any, output: Any):
        for param_group in self._all_gather_dtype_to_param_group.values():
            output = param_group.post_forward(module, input, output)
        return output

    def post_backward(self, *unused: Any):
        for param_group in self._all_gather_dtype_to_param_group.values():
            param_group.post_backward(*unused)

    def pre_backward(self, default_prefetch: bool, *unused: Any):
        for param_group in self._all_gather_dtype_to_param_group.values():
            param_group.pre_backward(default_prefetch, *unused)

    def finalize_backward(self):
        for param_group in self._all_gather_dtype_to_param_group.values():
            param_group.finalize_backward()

    @property
    def is_unsharded(self) -> bool:
        return next(iter(param_group.is_unsharded for param_group in self._all_gather_dtype_to_param_group.values()))

    @property
    def _reshard_after_forward(self):
        return next(
            iter(param_group._reshard_after_forward for param_group in self._all_gather_dtype_to_param_group.values())
        )

    @_reshard_after_forward.setter
    def _reshard_after_forward(self, value):
        for param_group in self._all_gather_dtype_to_param_group.values():
            param_group._reshard_after_forward = value
