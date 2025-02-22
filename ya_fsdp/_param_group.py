import contextlib
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, cast

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import _get_device_handle
from torch.distributed.fsdp._common_utils import _named_parameters_with_duplicates
from torch.profiler import record_function
from torch.utils._pytree import tree_flatten, tree_unflatten
from torch.utils.hooks import RemovableHandle

from ._api import MixedPrecisionPolicy, ShardedStateDictConfig, StateDictConfig, StateDictType
from ._collectives import all_gather, reduce_scatter
from ._common import FSDPMeshInfo, TrainingState
from ._param import ParamModuleInfo, ShardedState, YaFSDPParam

if TYPE_CHECKING:
    try:
        import yccl
    except ImportError:
        yccl = None

logger = logging.getLogger("ya_fsdp")

_ModuleToHandleDict = Dict[nn.Module, RemovableHandle]  # for state dict


class YaFSDPBufferContext:
    """This has the buffer state for all-gather / reduce-scatter ops shared across YaFSDP parameter groups."""

    def lazy_init(self, buffer_size: int, dtype: torch.dtype, device: torch.device, yccl_handle: Callable = None):
        if yccl_handle is not None and dtype != torch.bfloat16:
            raise RuntimeError("YCCL requires param_dtype and reduce_dtype to be bfloat16")
        self.buffer = (
            torch.empty(buffer_size, dtype=dtype, device=device) if yccl_handle is None else yccl_handle(buffer_size)
        )
        self.owner: Optional[YaFSDPParamGroup] = None
        self.release_event: Optional[torch.Event] = None


class YaFSDPCommContext:
    """This has the communication state shared across YaFSDP states/parameter groups."""

    def lazy_init(self, device: torch.device, yccl_handle: Optional["yccl.Handle"]):
        self.device_handle = _get_device_handle(device.type)
        high_priority = -1
        # All-gather stream allows overlapping next all-gather with current
        # forward compute
        self.all_gather_stream = self.device_handle.Stream(priority=high_priority)
        # Reduce-scatter stream gives separate execution "thread" for post-
        # backward logic like pre/post-gradient division and reduce-scatter
        self.reduce_scatter_stream = self.all_gather_stream
        self.post_forward_order: List[YaFSDPParamGroup] = []  # will cause ref cycles
        self.yccl_handle: Optional["yccl.Handle"] = yccl_handle

    def get_all_gather_stream(self, training_state: TrainingState) -> torch.Stream:
        if training_state in (TrainingState.FORWARD, TrainingState.PRE_BACKWARD):
            return self.all_gather_stream
        current_stream = self.device_handle.current_stream()
        return current_stream


class YaFSDPParamGroup:
    """This class represents a parameter group to communicate together."""

    _orig_dtype: torch.dtype
    _padded_unsharded_data_size: int
    _unsharded_data_offsets: list[int]
    _padded_sharded_param_data: torch.Tensor
    _padded_sharded_param_data_param_dtype: torch.Tensor
    _padded_sharded_param_grad: torch.Tensor
    _unsharded_param_data: torch.Tensor
    _unsharded_param_grad: torch.Tensor
    _unsharded_param_grad_reduce_dtype: Optional[torch.Tensor]

    def __init__(
        self,
        params: List[nn.Parameter],
        modules: Tuple[nn.Module, ...],
        mesh_info: FSDPMeshInfo,
        post_forward_mesh_info: Optional[FSDPMeshInfo],
        device: torch.device,
        mp_policy: MixedPrecisionPolicy,
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
        self.training_state = TrainingState.IDLE
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

        self._data_buffer_ctx = YaFSDPBufferContext()
        self._grad_buffer_ctx = YaFSDPBufferContext() if param_group_requires_grad else None
        self._reduce_dtype_grad_buffer_ctx = (
            YaFSDPBufferContext() if param_group_requires_grad and self._reduce_dtype is not None else None
        )

        self._state_dict_type: StateDictType = StateDictType.SHARDED_STATE_DICT
        self._state_dict_config: StateDictConfig = ShardedStateDictConfig()

        shard_world_size = self.mesh_info.shard_mesh_size

        unsharded_data_numels = [fsdp_param.param_data.numel() for fsdp_param in self.fsdp_params]
        self._unsharded_data_offsets = [0, *torch.cumsum(torch.tensor(unsharded_data_numels[:-1]), 0).tolist()]

        padded_unsharded_data_size = sum(unsharded_data_numels)
        divider = shard_world_size * 8
        if padded_unsharded_data_size % divider != 0:
            padded_unsharded_data_size += divider - padded_unsharded_data_size % divider
        self._padded_unsharded_data_size = padded_unsharded_data_size

        self._padded_sharded_param_data = torch.empty(
            padded_unsharded_data_size // shard_world_size, dtype=self._orig_dtype, device=device
        )
        self._padded_sharded_param_data_param_dtype = (
            self._padded_sharded_param_data.to(self._param_dtype) if self._param_dtype is not None else None
        )
        self._padded_sharded_param_grad = (
            torch.empty_like(self._padded_sharded_param_data) if param_group_requires_grad else None
        )

        padded_unsharded_data = torch.empty(padded_unsharded_data_size, dtype=self._orig_dtype, device=device)
        assert len(self.fsdp_params) < (max_indices_dtype_value := torch.iinfo((indices_dtype := torch.uint16)).max)
        padded_unsharded_data_indices = torch.full_like(
            padded_unsharded_data, fill_value=max_indices_dtype_value, dtype=indices_dtype
        )
        for i, (fsdp_param, offset, numel) in enumerate(
            zip(self.fsdp_params, self._unsharded_data_offsets, unsharded_data_numels)
        ):
            padded_unsharded_data[offset : offset + numel] = fsdp_param.param_data.view(-1)
            padded_unsharded_data_indices[offset : offset + numel] = i

        shard_rank = self.mesh_info.shard_mesh_rank

        self._padded_sharded_param_data.copy_(torch.chunk(padded_unsharded_data, shard_world_size)[shard_rank])
        if self._padded_sharded_param_data_param_dtype is not None:
            self._padded_sharded_param_data_param_dtype.copy_(self._padded_sharded_param_data)

        padded_sharded_data_indices = torch.chunk(padded_unsharded_data_indices, shard_world_size)[shard_rank]

        sharded_data_numels = [
            padded_sharded_data_indices.eq(index).sum().item() for index, _ in enumerate(self.fsdp_params)
        ]
        sharded_data_offsets = [0, *torch.cumsum(torch.tensor(sharded_data_numels[:-1]), 0).tolist()]

        for fsdp_param, offset, numel in zip(self.fsdp_params, sharded_data_offsets, sharded_data_numels):
            fsdp_param.init_sharded_param(
                self._padded_sharded_param_data.narrow(0, offset, numel),
                self._padded_sharded_param_grad.narrow(0, offset, numel)
                if self._padded_sharded_param_grad is not None
                else None,
            )

    # Initialization #
    def _init_unsharded_params(self):
        self._unsharded_param_data = self._data_buffer_ctx.buffer.narrow(0, 0, self._padded_unsharded_data_size)
        self._unsharded_param_grad = (
            self._grad_buffer_ctx.buffer.narrow(0, 0, self._padded_unsharded_data_size)
            if self._grad_buffer_ctx is not None
            else None
        )
        self._unsharded_param_grad_reduce_dtype = (
            self._reduce_dtype_grad_buffer_ctx.buffer.narrow(0, 0, self._padded_unsharded_data_size)
            if self._reduce_dtype_grad_buffer_ctx is not None
            else None
        )

        for fsdp_param, offset in zip(self.fsdp_params, self._unsharded_data_offsets):
            fsdp_param.init_unsharded_param(self._unsharded_param_data, self._unsharded_param_grad, offset)

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

    def lazy_init(self):
        # Lazy init should be idempotent
        if not hasattr(self.comm_ctx, "device_handle"):
            self.comm_ctx.device_handle = _get_device_handle(self.device.type)
        self._validate_no_meta_params()
        self._register_state_dict_hooks()
        self._init_unsharded_params()
        if (yccl_handle := self.comm_ctx.yccl_handle) is not None:
            if self._reduce_dtype is not None:
                raise NotImplementedError("YCCL requires param_dtype and reduce_dtype to be the same.")
            self._padded_sharded_param_data_param_dtype = yccl_handle.add_all_gather_input_buffer(
                self._padded_sharded_param_data_param_dtype.numel()
            ).copy_(self._padded_sharded_param_data_param_dtype)

    # Runtime #
    def unshard(self):
        if self._all_gather_event is not None:  # already called, pending wait
            return
        if self.is_unsharded:
            return  # no-op
        if not self.unshard_in_backward and self.training_state == TrainingState.PRE_BACKWARD:
            return
        # if self._reshard_after_forward_event is not None:
        #     # Resharded parameter data is allocated in the default stream and
        #     # used in the all-gather streams
        #     self._wait_all_gather_stream_on_event(self._reshard_after_forward_event)
        #     self._reshard_after_forward_event = None
        with record_function(self._with_fqn("YaFSDP::all_gather")):
            self._all_gather_event = all_gather(
                self,
                self._padded_sharded_param_data,
                self._padded_sharded_param_data_param_dtype,
                self._unsharded_param_data,
                self._data_buffer_ctx,
                self._all_gather_process_group,
                self.comm_ctx.get_all_gather_stream(self.training_state),
                self._param_dtype,
                self.device_handle,
                self.comm_ctx.yccl_handle,
            )

    def wait_for_unshard(self):
        if self._all_gather_event:
            self.device_handle.current_stream().wait_event(self._all_gather_event)
        self._all_gather_event = None
        self._to_unsharded()

    def reshard(self):
        if self.training_state == TrainingState.FORWARD:
            if not self._reshard_after_forward:
                return
            if self._use_post_forward_mesh:
                raise NotImplementedError("YaFSDP does not support post-forward mesh resharding yet")
                # self._to_sharded_post_forward()
                # self._reshard_after_forward_event = torch.Event()
                # self._reshard_after_forward_event.record()
                # return
        self._to_sharded()
        self._data_buffer_ctx.release_event = self.device_handle.current_stream().record_event()
        self._data_buffer_ctx.owner = None

    def pre_forward(
        self, module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        logger.debug("%s", self._with_fqn("YaFSDP::pre_forward"))
        with record_function(self._with_fqn("YaFSDP::pre_forward")):
            self.training_state = TrainingState.FORWARD
            self.unshard()
            args, kwargs, unsharded_params = self._register_post_backward_hook(args, kwargs)
            for fsdp_param, unsharded_param in unsharded_params.items():
                fsdp_param._unsharded_params_for_backward.append(unsharded_param)
            self.wait_for_unshard()
            return args, kwargs

    def post_forward(self, module: nn.Module, input: Any, output: Any):
        logger.debug("%s", self._with_fqn("YaFSDP::post_forward"))
        with record_function(self._with_fqn("YaFSDP::post_forward")):
            self.reshard()
            if torch.is_grad_enabled():
                self._record_post_forward()
            self.training_state = TrainingState.IDLE
            return output

    def _record_post_forward(self) -> None:
        # Since a group has one pre-backward unshard for each forward call
        # before the backward, we record each usage (with multiplicity)
        post_forward_index = len(self.comm_ctx.post_forward_order)
        self.comm_ctx.post_forward_order.append(self)
        self._post_forward_indices.append(post_forward_index)

    def pre_backward(self):
        if self.training_state == TrainingState.PRE_BACKWARD:
            return
        logger.debug("%s", self._with_fqn("YaFSDP::pre_backward"))
        with record_function(self._with_fqn("YaFSDP::pre_backward")):
            self.training_state = TrainingState.PRE_BACKWARD
            self.unshard()
            self.wait_for_unshard()
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
        self.training_state = TrainingState.POST_BACKWARD
        with record_function(self._with_fqn("YaFSDP::post_backward_reshard")):
            if not self.reduce_grads:
                if self.reshard_after_backward:
                    self.reshard()
                self._backward_prefetch()
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
            self._backward_prefetch()
        if self._grad_buffer_ctx is None:
            assert len(fsdp_params_with_grad) == 0
            return
        if len(fsdp_params_with_grad) != 0:
            with record_function(self._with_fqn("YaFSDP::post_backward_reduce")):
                self.post_reduce_event, grad_buffer_release_event = reduce_scatter(
                    self,
                    fsdp_params_with_grad,
                    self._padded_sharded_param_grad,
                    self._unsharded_param_grad,
                    self._unsharded_param_grad_reduce_dtype,
                    self._reduce_dtype_grad_buffer_ctx,
                    self._reduce_scatter_process_group,
                    self.comm_ctx.reduce_scatter_stream,
                    self._orig_dtype,
                    self._param_dtype,
                    self._reduce_dtype,
                    self.device_handle,
                    self.mp_policy.bit32_acc_for_bit16_reduce_scatter,
                    self.comm_ctx.yccl_handle,
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
        self._post_forward_indices.clear()
        for fsdp_param in self.fsdp_params:
            if fsdp_param.sharded_param.requires_grad:
                assert len(fsdp_param._unsharded_params_for_backward) == 0

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
        with record_function(
            f"YaFSDP::{pass_type}_prefetch for {target_fqn}"
        ), target_fsdp_param_group.use_training_state(training_state):
            target_fsdp_param_group.unshard()

    # Utilities #
    def _to_sharded(self):
        if not self.is_sharded:
            for fsdp_param in self.fsdp_params:
                fsdp_param.to_sharded()
            self._sharded_state = ShardedState.SHARDED

    # def _to_sharded_post_forward(self):
    #     if not self.is_sharded_post_forward:
    #         for fsdp_param in self.fsdp_params:
    #             fsdp_param.to_sharded_post_forward()
    #         self._sharded_state = ShardedState.SHARDED_POST_FORWARD

    def _to_unsharded(self):
        for fsdp_param in self.fsdp_params:
            fsdp_param.to_unsharded(self.training_state)
        if not self.is_unsharded:
            self._sharded_state = ShardedState.UNSHARDED

    @property
    def is_sharded(self) -> bool:
        return self._sharded_state == ShardedState.SHARDED

    @property
    def is_sharded_post_forward(self) -> bool:
        return self._sharded_state == ShardedState.SHARDED_POST_FORWARD

    @property
    def is_unsharded(self) -> bool:
        return self._sharded_state == ShardedState.UNSHARDED

    @contextlib.contextmanager
    def use_training_state(self, training_state: TrainingState):
        old_training_state = self.training_state
        self.training_state = training_state
        try:
            yield
        finally:
            self.training_state = old_training_state

    # Hook Registration #
    def _register_post_backward_hook(
        self,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
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
        unsharded_params = {
            fsdp_param: unsharded_param
            for fsdp_param, unsharded_param in zip(fsdp_params_with_grads, unsharded_params, strict=True)
        }
        if len(inp_tensors) == 0:
            return args, kwargs, unsharded_params  # no tensors that require gradients
        for inp_tensor_idx, inp_tensor in zip(inp_tensor_indices, inp_tensors):
            args_kwargs_list[inp_tensor_idx] = inp_tensor
        args_list = args_kwargs_list[: len(args_list)]
        kwargs_list = args_kwargs_list[len(args_list) :]
        args = tree_unflatten(args_list, args_spec)
        kwargs = tree_unflatten(kwargs_list, kwargs_spec)
        return args, kwargs, unsharded_params

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
            if self._state_dict_type is StateDictType.FULL_STATE_DICT and self._state_dict_config.rank0_only:
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
    def _use_post_forward_mesh(self) -> bool:
        return self._reshard_after_forward and self.mesh_info != self.post_forward_mesh_info

    @property
    def _all_gather_process_group(self) -> dist.ProcessGroup:
        assert not self.is_sharded_post_forward
        mesh_info = cast(FSDPMeshInfo, self.post_forward_mesh_info) if self.is_sharded_post_forward else self.mesh_info
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
        fsdp_params_with_grads = [
            fsdp_param for fsdp_param in ctx.param_group.fsdp_params if fsdp_param.sharded_param.requires_grad
        ]
        unsharded_param_grads, inp_grads = grads[: len(fsdp_params_with_grads)], grads[len(fsdp_params_with_grads) :]
        # it's required that all (unsharded) params which require grad receive a gradient
        for fsdp_param, unsharded_param_grad in zip(fsdp_params_with_grads, unsharded_param_grads, strict=True):
            if unsharded_param_grad is None:
                raise ValueError(
                    f"{fsdp_param._param_fqn} requires grad, got unsharded during forward"
                    ", but got no gradient after backward."
                )
            if fsdp_param._unsharded_param.grad is None:
                fsdp_param._unsharded_param_grad.copy_(unsharded_param_grad)
                fsdp_param._unsharded_param.grad = fsdp_param._unsharded_param_grad
            else:
                fsdp_param._unsharded_param.grad.add_(unsharded_param_grad)
        ctx.param_group.post_backward()
        return (
            None,
            *(None for _ in unsharded_param_grads),
            *inp_grads,
        )
