import functools
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
from torch._logging import warning_once
from torch.autograd import Variable
from torch.autograd.graph import _MultiHandle
from torch.distributed._composable.fsdp._fsdp_common import _cast_fp_tensor
from torch.distributed._composable_state import _get_module_state, _insert_module_state, _State
from torch.distributed.device_mesh import _get_device_handle
from torch.distributed.utils import _to_kwargs
from torch.utils._pytree import tree_flatten, tree_map

from ._api import MixedPrecisionPolicy
from ._common import TrainingState
from ._param_group import YaFSDPCommContext, YaFSDPParamGroup

if TYPE_CHECKING:
    from ._param import YaFSDPParam

    try:
        import yccl
    except ImportError:
        yccl = None

logger = logging.getLogger("ya_fsdp")


class YaFSDPStateContext:
    """This has state shared across YaFSDP states."""

    def __init__(self) -> None:
        # All YaFSDP states in the root state's module tree
        self.all_states: List[YaFSDPState] = []
        # Iteration's forward root runs the once-per-forward logic; this root
        # may not be the overall root set by lazy initialization in cases where
        # only a submodule runs forward (e.g. encoder-only for eval)
        self.iter_forward_root: Optional[YaFSDPState] = None
        # Final callback should only be queued once per backward
        self.post_backward_final_callback_queued: bool = False
        # Whether to finalize backward in this backward's final callback
        self.is_last_backward: bool = True
        # Optional user-provided event recorded after optimizer for the
        # all-gather streams to wait on in the root pre-forward
        self.post_optim_event: Optional[torch.Event] = None


class YaFSDPState(_State):
    def __init__(self) -> None:
        super().__init__()
        self._fsdp_param_group: Optional[YaFSDPParamGroup] = None
        self._is_root: Optional[bool] = None  # root set during lazy init
        self._state_ctx = YaFSDPStateContext()
        self._comm_ctx = YaFSDPCommContext()
        self._training_state: TrainingState = TrainingState.IDLE
        self._modules_to_run_forward: Set[nn.Module] = set()

    # Define a separate init since `__init__` is called in the contract
    def init(
        self,
        modules: Tuple[nn.Module, ...],
        device: torch.device,
        mp_policy: MixedPrecisionPolicy,
    ) -> None:
        for module in modules:
            _insert_module_state(module, self)
        self._modules = modules
        self._device = device
        self._device_handle = _get_device_handle(device.type)
        self._mp_policy = mp_policy
        if len(modules) == 1:
            self._pre_forward_hook_handle = modules[0].register_forward_pre_hook(
                self._pre_forward, prepend=True, with_kwargs=True
            )
            self._post_forward_hook_handle = modules[0].register_forward_hook(self._post_forward, prepend=False)
        else:
            hook_handle = _register_group_forward_hooks(
                modules,
                self._pre_forward,
                self._post_forward,
                self._modules_to_run_forward,
            )
            self._pre_forward_hook_handle = hook_handle
            self._post_forward_hook_handle = hook_handle

    def _root_pre_forward(
        self, module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        self._lazy_init()
        if self._state_ctx.iter_forward_root is not None:
            return args, kwargs
        logger.debug("YaFSDP::root_pre_forward")
        self._state_ctx.iter_forward_root = self
        with torch.profiler.record_function("YaFSDP::root_pre_forward"):
            # Wait for optimizer before implicitly prefetched all-gathers
            if (event := self._state_ctx.post_optim_event) is not None:
                self._comm_ctx.all_gather_stream.wait_event(event)
                self._state_ctx.post_optim_event = None
            else:
                current_stream = self._device_handle.current_stream()
                self._comm_ctx.all_gather_stream.wait_stream(current_stream)
            if self._device.type == "cuda":
                with torch.profiler.record_function("YaFSDP::inputs_to_device"):
                    args_tuple, kwargs_tuple = _to_kwargs(args, kwargs, self._device, False)
                args, kwargs = args_tuple[0], kwargs_tuple[0]
        return args, kwargs

    def _lazy_init(
        self,
        num_data_buffers: int = 2,
        num_grad_buffers: int = 2,
        meta_grad_buffers: bool = False,
        yccl_handle: Optional["yccl.Handle"] = None,
    ) -> None:
        """
        Lazy initialization represents when all modules' parallelisms have
        finalized (e.g. YaFSDP has been applied to all desired modules). This
        means that we can determine which state is the root, and we do so by
        the 1st state to run forward.
        """
        if self._is_root is not None:
            return  # no-op: already initialized
        self._is_root = True
        if len(self._modules) > 1:
            raise RuntimeError(f"YaFSDP requires a single root module but got {self._modules}")
        root_module = self._modules[0]
        visited_states: Set[YaFSDPState] = set()
        all_states: List[YaFSDPState] = []
        for module_name, module in root_module.named_modules():
            if (state := _get_module_fsdp_state(module)) is None:
                continue
            if module is not root_module:
                if state not in visited_states and state._is_root is not None:
                    raise RuntimeError(
                        "YaFSDP state has already been lazily initialized for "
                        f"{module_name}\nFSDP requires running forward through "
                        "the root module first"
                    )
                state._is_root = False
            all_states.append(state)
            visited_states.add(state)
        for prev_state, state in zip([None, *all_states[:-1]], all_states):
            if prev_state is not None and state is prev_state:
                continue
            self._state_ctx.all_states.append(state)
        assert len(visited_states) == len(self._state_ctx.all_states)
        if self._fsdp_param_group and self._state_ctx.all_states > 1:
            raise RuntimeError("YaFSDP requires root module to be the only sharded module or to have no parameters.")
        self._init_fqns()
        self._init_shared_state(num_data_buffers, num_grad_buffers, meta_grad_buffers, yccl_handle)
        # Run parameter group lazy inits after initializing FQNs for improved
        # error messages
        for state in self._state_ctx.all_states:
            if state._fsdp_param_group:
                state._fsdp_param_group.lazy_init()

    def _init_shared_state(
        self,
        num_data_buffers: int,
        num_grad_buffers: int,
        meta_grad_buffers: bool,
        yccl_handle: Optional["yccl.Handle"],
    ) -> None:
        self._comm_ctx.lazy_init(self._device, yccl_handle)
        for state in self._state_ctx.all_states:
            state._state_ctx = self._state_ctx
            state._comm_ctx = self._comm_ctx
            if fsdp_param_group := state._fsdp_param_group:
                fsdp_param_group.comm_ctx = self._comm_ctx

        param_groups = [
            state._fsdp_param_group for state in self._state_ctx.all_states if state._fsdp_param_group is not None
        ]

        buffer_size = max(param_group._padded_unsharded_data_size for param_group in param_groups)
        for index, param_group in enumerate(param_groups):
            if (
                not param_group._reshard_after_forward
                and param_group not in param_groups[-2:]
                and num_data_buffers < len(param_groups)
            ):
                raise ValueError(
                    "YaFSDP with reshard_after_forward requires number of data buffers to be no less than number of"
                    f" parameter groups, but got {num_data_buffers=}"
                )
            if index < num_data_buffers:
                param_group._data_buffer_ctx.lazy_init(
                    buffer_size,
                    self._mp_policy.param_dtype,
                    self._device,
                    yccl_handle=yccl_handle.add_all_gather_output_buffer if yccl_handle is not None else None,
                )
            if index < num_grad_buffers:
                param_group._grad_buffer_ctx.lazy_init(
                    buffer_size,
                    self._mp_policy.param_dtype,
                    self._device if not meta_grad_buffers else torch.device("meta"),
                    yccl_handle=yccl_handle.add_reduce_scatter_buffer if yccl_handle is not None else None,
                )
            param_group._data_buffer_ctx = param_groups[index % num_data_buffers]._data_buffer_ctx
            param_group._grad_buffer_ctx = param_groups[index % num_grad_buffers]._grad_buffer_ctx

        if self._mp_policy.reduce_dtype is not None:
            for index, param_group in enumerate(param_groups):
                if index == 0:
                    param_group._reduce_dtype_grad_buffer_ctx.lazy_init(
                        buffer_size, self._mp_policy.reduce_dtype, self._device
                    )
                param_group._reduce_dtype_grad_buffer_ctx = param_groups[0]._reduce_dtype_grad_buffer_ctx

    def _init_fqns(self) -> None:
        """Sets module and parameter FQN attributes for debugging."""
        assert self._is_root
        root_module = self._modules[0]
        param_to_fsdp_param: Dict[nn.Parameter, YaFSDPParam] = {}
        module_to_fsdp_param_group: Dict[nn.Module, YaFSDPParamGroup] = {}
        for state in self._state_ctx.all_states:
            if fsdp_param_group := state._fsdp_param_group:
                for fsdp_param in fsdp_param_group.fsdp_params:
                    param_to_fsdp_param[fsdp_param.sharded_param] = fsdp_param
                for module in fsdp_param_group.modules:
                    module_to_fsdp_param_group[module] = fsdp_param_group
        for param_name, param in root_module.named_parameters():
            if param in param_to_fsdp_param:
                param_to_fsdp_param[param]._param_fqn = param_name
        for module_name, module in root_module.named_modules():
            if module in module_to_fsdp_param_group:
                module_fqn = module_to_fsdp_param_group[module]._module_fqn
                if module_fqn is None:
                    module_to_fsdp_param_group[module]._module_fqn = module_name
                else:
                    assert isinstance(module_fqn, str), f"{module_fqn}"
                    module_fqn += f", {module_name}"
                    module_to_fsdp_param_group[module]._module_fqn = module_fqn

    def _pre_forward(
        self, module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        # When composing with module-hook-based activation checkpointing, the
        # the pre-backward hook is responsible for the unshard
        if self._training_state == TrainingState.PRE_BACKWARD:
            return args, kwargs
        self._training_state = TrainingState.FORWARD
        args, kwargs = self._root_pre_forward(module, args, kwargs)
        if self._mp_policy.cast_forward_inputs and self._mp_policy.param_dtype:
            with torch.profiler.record_function("YaFSDP::cast_forward_inputs"):
                cast_fn = functools.partial(_cast_fp_tensor, self._mp_policy.param_dtype)
                args, kwargs = tree_map(cast_fn, args), tree_map(cast_fn, kwargs)
        if self._fsdp_param_group:
            args, kwargs = self._fsdp_param_group.pre_forward(module, args, kwargs)
        return args, kwargs

    def _post_forward(self, module: nn.Module, input: Any, output: Any) -> Any:
        # When composing with module-hook-based activation checkpointing, the
        # post-backward hook is responsible for the reshard
        if self._training_state == TrainingState.PRE_BACKWARD:
            return output
        if self._fsdp_param_group:
            output = self._fsdp_param_group.post_forward(module, input, output)
        output = self._register_pre_backward_hook(output)
        self._training_state = TrainingState.IDLE
        if self._state_ctx.iter_forward_root is self:
            self._state_ctx.iter_forward_root = None
        if self._mp_policy.output_dtype is not None:
            with torch.profiler.record_function("YaFSDP::cast_forward_outputs"):
                output = tree_map(
                    functools.partial(_cast_fp_tensor, self._mp_policy.output_dtype),
                    output,
                )
        return output

    def _pre_backward(self, grad: torch.Tensor) -> torch.Tensor:
        self._training_state = TrainingState.PRE_BACKWARD
        self._register_root_post_backward_final_callback()
        if self._fsdp_param_group:
            self._fsdp_param_group.pre_backward()
        return grad

    def _root_post_backward_final_callback(self) -> None:
        logger.debug("YaFSDP::root_post_backward")
        with torch.profiler.record_function("YaFSDP::root_post_backward_callback"):
            for state in self._state_ctx.all_states:
                fsdp_param_group = state._fsdp_param_group
                if fsdp_param_group and fsdp_param_group.is_unsharded and fsdp_param_group.reshard_after_backward:
                    # Run post-backward in case forward inputs did not require
                    # gradient so the autograd backward did not run
                    fsdp_param_group.post_backward()
                state._training_state = TrainingState.IDLE
                if fsdp_param_group:
                    fsdp_param_group._training_state = TrainingState.IDLE
                if self._state_ctx.is_last_backward:
                    state._finalize_backward()
            self._comm_ctx.post_forward_order.clear()
            self._state_ctx.post_backward_final_callback_queued = False
            if self._comm_ctx.yccl_handle is not None:
                self._comm_ctx.yccl_handle.process_profiling_events()

    def _finalize_backward(self) -> None:
        if self._modules_to_run_forward:
            msg = (
                f"{len(self._modules_to_run_forward)} of the {len(self._modules)} "
                f"modules passed to fully_shard did not run forward before backward, "
                "which is error-prone since FSDP post-forward/pre-backward logic "
                "will not run for these modules. We recommend passing only modules "
                "that run forward together. Modules that did not run forward: "
                f"{list(self._modules_to_run_forward)}"
            )
            warning_once(logger, msg, stacklevel=2)
            # Clear since we want the next forward to run
            self._modules_to_run_forward.clear()
        if self._fsdp_param_group:
            self._fsdp_param_group.finalize_backward()

    def _register_pre_backward_hook(self, output: Any) -> Any:
        if not torch.is_grad_enabled():
            return output
        flat_outputs, _ = tree_flatten(output)
        for t in flat_outputs:
            if torch.is_tensor(t) and t.requires_grad:
                t.register_hook(self._pre_backward)
        return output

    def _register_root_post_backward_final_callback(self):
        if self._state_ctx.post_backward_final_callback_queued:
            return
        self._state_ctx.post_backward_final_callback_queued = True
        Variable._execution_engine.queue_callback(self._root_post_backward_final_callback)


def _get_module_fsdp_state(module: nn.Module) -> Optional[YaFSDPState]:
    state = _get_module_state(module)
    if isinstance(state, YaFSDPState):
        return state
    return None


def _register_group_forward_hooks(
    modules: Sequence[nn.Module],
    pre_hook: Callable,
    post_hook: Callable,
    modules_to_run: Set[nn.Module],
):
    """
    Registers group forward pre and post-hooks. The pre-hook runs upon the
    first module pre-forward, and the post-hook runs upon the last. If at least
    one module does not run forward, then the post-hook does not run.
    """
    modules_set = set(modules)

    @functools.wraps(pre_hook)
    def wrapped_pre_hook(*args: Any, **kwargs: Any):
        if len(modules_to_run) == 0:  # first to run
            modules_to_run.update(modules_set)
            return pre_hook(*args, **kwargs)

    def get_wrapped_post_hook(module: nn.Module):
        @functools.wraps(post_hook)
        def wrapped_post_hook(*args: Any, **kwargs: Any):
            modules_to_run.discard(module)
            if len(modules_to_run) == 0:
                return post_hook(*args, **kwargs)

        return wrapped_post_hook

    pre_handles = [
        module.register_forward_pre_hook(wrapped_pre_hook, prepend=True, with_kwargs=True) for module in modules
    ]
    post_handles = [
        module.register_forward_hook(get_wrapped_post_hook(module), prepend=False, always_call=True)
        for module in modules
    ]
    return _MultiHandle(tuple(pre_handles + post_handles))
