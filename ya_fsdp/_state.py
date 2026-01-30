import functools
import logging
from collections.abc import Callable, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    TypeVar,
)

import torch
import torch.nn as nn
from torch._logging import warning_once
from torch.autograd import Variable
from torch.autograd.graph import _MultiHandle
from torch.distributed._composable_state import (
    _get_module_state,
    _insert_module_state,
    _State,
)
from torch.distributed.device_mesh import _get_device_handle
from torch.distributed.utils import _to_kwargs
from torch.utils._pytree import tree_flatten, tree_map

from ._api import MixedPrecisionPolicy
from ._common import TrainingState, _cast_fp_tensor
from ._param_group import YaFSDPBufferContext, YaFSDPCommContext, YaFSDPParamGroup

if TYPE_CHECKING:
    from ._param import YaFSDPParam

    try:
        import yccl
    except ImportError:
        pass

logger = logging.getLogger("ya_fsdp")


class YaFSDPStateContext:
    def __init__(self) -> None:
        # All YaFSDP states in the root state's module tree
        self.all_states: list[YaFSDPState] = []
        # Iteration's forward root runs the once-per-forward logic; this root
        # may not be the overall root set by lazy initialization in cases where
        # only a submodule runs forward (e.g. encoder-only for eval)
        self.iter_forward_root: YaFSDPState | None = None
        # Final callback should only be queued once per backward
        self.post_backward_final_callback_queued: bool = False
        # Whether to finalize backward in this backward's final callback
        self.is_last_backward: bool = True
        # Optional user-provided event recorded after optimizer for the
        # all-gather streams to wait on in the root pre-forward
        self.post_optim_event: torch.Event | None = None


class YaFSDPState(_State):
    def __init__(self) -> None:
        super().__init__()
        self._fsdp_param_group: YaFSDPParamGroup | None = None
        self._is_root: bool | None = None  # root set during lazy init
        self._state_ctx = YaFSDPStateContext()
        self._comm_ctx = YaFSDPCommContext()
        self._training_state: TrainingState = TrainingState.IDLE
        self._states_to_forward_prefetch: list[YaFSDPState] = []
        self._states_to_backward_prefetch: list[YaFSDPState] = []
        self._modules_to_run_forward: set[nn.Module] = set()

    # Define a separate init since `__init__` is called in the contract
    def init(
        self,
        modules: tuple[nn.Module, ...],
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
            self._post_forward_hook_handle = modules[0].register_forward_hook(
                self._post_forward, prepend=False
            )
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
        self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
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
                    args_tuple, kwargs_tuple = _to_kwargs(
                        args, kwargs, self._device, False
                    )  # same as DDP
                args, kwargs = args_tuple[0], kwargs_tuple[0]
        return args, kwargs

    def _lazy_init(
        self,
        allow_no_grad_reduce: bool = False,
        yccl_handle: Optional["yccl.Handle"] = None,
        param_group_to_data_buffer_ctx_idx: dict[YaFSDPParamGroup, int] | None = None,
        param_group_to_grad_buffer_ctx_idx: dict[YaFSDPParamGroup, int] | None = None,
        data_buffer_ctx_idx_to_yccl_handle: dict[int, "yccl.Handle"] | None = None,
        grad_buffer_ctx_idx_to_yccl_handle: dict[int, "yccl.Handle"] | None = None,
    ) -> None:
        if self._is_root is not None:
            return  # no-op: already initialized
        self._is_root = True
        if len(self._modules) > 1:
            raise RuntimeError(
                f"YaFSDP requires a single root module but got {self._modules}"
            )
        root_module = self._modules[0]
        visited_states: set[YaFSDPState] = set()
        for module_name, module in root_module.named_modules():
            if (state := _get_module_fsdp_state(module)) is None:
                continue
            if module is not root_module:
                if state not in visited_states and state._is_root is not None:
                    raise RuntimeError(
                        "YaFSDP state has already been lazily initialized for "
                        f"{module_name}\nYaFSDP requires running forward through "
                        "the root module first"
                    )
                state._is_root = False
            if state not in visited_states:
                self._state_ctx.all_states.append(state)
            visited_states.add(state)
        self._init_fqns()
        self._init_shared_state(
            allow_no_grad_reduce,
            yccl_handle,
            param_group_to_data_buffer_ctx_idx,
            param_group_to_grad_buffer_ctx_idx,
            data_buffer_ctx_idx_to_yccl_handle,
            grad_buffer_ctx_idx_to_yccl_handle,
        )
        self._validate_shared_state()
        # Run parameter group lazy inits after initializing FQNs for improved
        # error messages
        for state in self._state_ctx.all_states:
            if state._fsdp_param_group:
                state._fsdp_param_group.lazy_init()

    def _init_shared_state(
        self,
        allow_no_grad_reduce: bool,
        yccl_handle: Optional["yccl.Handle"],
        param_group_to_data_buffer_ctx_idx: dict[YaFSDPParamGroup, int] | None,
        param_group_to_grad_buffer_ctx_idx: dict[YaFSDPParamGroup, int] | None,
        data_buffer_ctx_idx_to_yccl_handle: dict[int, "yccl.Handle"] | None,
        grad_buffer_ctx_idx_to_yccl_handle: dict[int, "yccl.Handle"] | None,
    ) -> None:
        self._comm_ctx.lazy_init(self._device)
        for state in self._state_ctx.all_states:
            state._state_ctx = self._state_ctx
            state._comm_ctx = self._comm_ctx
            if fsdp_param_group := state._fsdp_param_group:
                fsdp_param_group.comm_ctx = self._comm_ctx

        param_groups = [
            state._fsdp_param_group
            for state in self._state_ctx.all_states
            if state._fsdp_param_group is not None
        ]

        no_reshard_after_forward = all(
            not param_group._reshard_after_forward for param_group in param_groups
        )
        data_buffer_ctx2ctx_using_param_groups, data_buffer_ctx2yccl_handle = (
            get_buffer_ctx2ctx_using_param_groups_map(
                param_groups,
                "_data_buffer_ctx",
                len(param_groups) if no_reshard_after_forward else 2,
                param_group_to_data_buffer_ctx_idx,
                data_buffer_ctx_idx_to_yccl_handle,
            )
        )
        grad_buffer_ctx2ctx_using_param_groups, grad_buffer_ctx2yccl_handle = (
            get_buffer_ctx2ctx_using_param_groups_map(
                (
                    param_groups_with_grads := [
                        param_group
                        for param_group in param_groups
                        if param_group._grad_buffer_ctx is not None
                    ]
                ),
                "_grad_buffer_ctx",
                len(param_groups_with_grads) if allow_no_grad_reduce else 2,
                param_group_to_grad_buffer_ctx_idx,
                grad_buffer_ctx_idx_to_yccl_handle,
            )
        )
        reduce_dtype_grad_buffer_ctx2ctx_using_param_groups, _ = (
            get_buffer_ctx2ctx_using_param_groups_map(
                [
                    param_group
                    for param_group in param_groups
                    if param_group._reduce_dtype_grad_buffer_ctx is not None
                ],
                "_reduce_dtype_grad_buffer_ctx",
                1,
            )
        )

        T = TypeVar("T")

        def max_across_dict(dict_list: list[dict[T, int]]) -> dict[T, int]:
            max_dict = {}
            for d in dict_list:
                for k, v in d.items():
                    if k in max_dict:
                        max_dict[k] = max(max_dict[k], v)
                    else:
                        max_dict[k] = v
            return max_dict

        for (
            data_buffer_ctx,
            ctx_using_param_groups,
        ) in data_buffer_ctx2ctx_using_param_groups.items():
            padded_unsharded_param_sizes = [
                param_group._padded_unsharded_param_size
                for param_group in ctx_using_param_groups
            ]
            buffer_size = (
                max
                if isinstance(next(iter(padded_unsharded_param_sizes)), int)
                else max_across_dict
            )(padded_unsharded_param_sizes)
            param_dtypes = {
                param_group._param_dtype or param_group._orig_dtype
                for param_group in ctx_using_param_groups
            }
            if len(param_dtypes) != 1:
                raise AssertionError(
                    f"YaFSDP expects uniform param dtype across YaFSDPParamGroups which share a data buffer but got {param_dtypes}"
                )
            param_dtype = next(iter(param_dtypes))
            data_buffer_ctx.lazy_init(
                buffer_size,
                param_dtype,
                self._device,
                yccl_handle=yccl_handle
                if data_buffer_ctx2yccl_handle is None
                else data_buffer_ctx2yccl_handle[data_buffer_ctx],
            )
            for param_group in ctx_using_param_groups:
                param_group._data_buffer_ctx = data_buffer_ctx
            # Do not reshard the last module across modules using the same data buffer
            # after forward since for training the parameters would be
            # freed and all-gathered immediately
            ctx_using_param_groups[-1].post_forward_mesh_info = None
        for (
            grad_buffer_ctx,
            ctx_using_param_groups,
        ) in grad_buffer_ctx2ctx_using_param_groups.items():
            padded_unsharded_param_sizes = [
                param_group._padded_unsharded_param_size
                for param_group in ctx_using_param_groups
            ]
            buffer_size = (
                max
                if isinstance(next(iter(padded_unsharded_param_sizes)), int)
                else max_across_dict
            )(padded_unsharded_param_sizes)
            param_dtypes = {
                param_group._param_dtype or param_group._orig_dtype
                for param_group in ctx_using_param_groups
            }
            if len(param_dtypes) != 1:
                raise AssertionError(
                    f"YaFSDP expects uniform param dtype across YaFSDPParamGroups which share a grad buffer but got {param_dtypes}"
                )
            param_dtype = next(iter(param_dtypes))
            grad_buffer_ctx.lazy_init(
                buffer_size,
                param_dtype,
                self._device,
                yccl_handle=yccl_handle
                if grad_buffer_ctx2yccl_handle is None
                else grad_buffer_ctx2yccl_handle[grad_buffer_ctx],
            )
            for param_group in ctx_using_param_groups:
                param_group._grad_buffer_ctx = grad_buffer_ctx
        for (
            reduce_dtype_grad_buffer_ctx,
            ctx_using_param_groups,
        ) in reduce_dtype_grad_buffer_ctx2ctx_using_param_groups.items():
            padded_unsharded_param_sizes = [
                param_group._padded_unsharded_param_size
                for param_group in ctx_using_param_groups
            ]
            buffer_size = (
                max
                if isinstance(next(iter(padded_unsharded_param_sizes)), int)
                else max_across_dict
            )(padded_unsharded_param_sizes)
            reduce_dtypes = {
                param_group._reduce_dtype for param_group in ctx_using_param_groups
            }
            if len(reduce_dtypes) != 1:
                raise AssertionError(
                    f"YaFSDP expects uniform reduce dtype across YaFSDPParamGroups which share a reduce dtype grad buffer but got {param_dtypes}"
                )
            reduce_dtype = next(iter(reduce_dtypes))
            reduce_dtype_grad_buffer_ctx.lazy_init(
                buffer_size,
                reduce_dtype,
                self._device,
            )
            for param_group in ctx_using_param_groups:
                param_group._reduce_dtype_grad_buffer_ctx = reduce_dtype_grad_buffer_ctx

    def _validate_shared_state(self) -> None:
        data_buffer_ctx2ctx_using_param_groups = {}
        grad_buffer_ctx2ctx_using_param_groups = {}
        for state in self._state_ctx.all_states:
            if (param_group := state._fsdp_param_group) is None:
                continue
            data_buffer_ctx2ctx_using_param_groups.setdefault(
                param_group._data_buffer_ctx, []
            ).append(param_group)
            if param_group._grad_buffer_ctx is None:
                continue
            grad_buffer_ctx2ctx_using_param_groups.setdefault(
                param_group._grad_buffer_ctx, []
            ).append(param_group)
        if (num_data_buffers := len(data_buffer_ctx2ctx_using_param_groups)) < min(
            2,
            num_param_groups := sum(
                map(len, data_buffer_ctx2ctx_using_param_groups.values())
            ),
        ):
            raise ValueError(
                "num_data_buffers must be no less than min(2, num_param_groups)"
                f" for correct overlap, but got {num_data_buffers=} {num_param_groups=}."
            )
        if (num_grad_buffers := len(grad_buffer_ctx2ctx_using_param_groups)) < min(
            2,
            num_param_groups := sum(
                map(len, grad_buffer_ctx2ctx_using_param_groups.values())
            ),
        ):
            raise ValueError(
                "num_grad_buffers must be no less than min(2, num_param_groups)"
                f" for correct overlap, but got {num_grad_buffers=} {num_param_groups=}."
            )
        for ctx_using_param_groups in data_buffer_ctx2ctx_using_param_groups.values():
            if (
                len(
                    param_groups_with_no_reshard_after_forward := [
                        param_group
                        for param_group in ctx_using_param_groups
                        if not param_group._reshard_after_forward
                    ]
                )
                > 1
            ):
                raise ValueError(
                    "Only one parameter group across groups which share a data buffer"
                    " is allowed to have no reshard after forward"
                    f", but got {param_groups_with_no_reshard_after_forward=}"
                )
            if (
                len(
                    param_groups_with_no_reshard_after_backward := [
                        param_group
                        for param_group in ctx_using_param_groups
                        if not param_group.reshard_after_backward
                    ]
                )
                > 1
            ):
                raise ValueError(
                    "Only one parameter group across groups which share a data buffer"
                    " is allowed to have no reshard after backward"
                    f", but got {param_groups_with_no_reshard_after_backward=}"
                )
        for ctx_using_param_groups in grad_buffer_ctx2ctx_using_param_groups.values():
            if (
                len(
                    param_groups_with_no_reduce_grads := [
                        param_group
                        for param_group in ctx_using_param_groups
                        if not param_group.reduce_grads
                    ]
                )
                > 1
            ):
                raise ValueError(
                    "Only one parameter group across groups which share a grad buffer"
                    "is allowed to have no grad reduce"
                    f", but got {param_groups_with_no_reduce_grads=}"
                )
        data_buffer_ctx2module_prefix: dict[YaFSDPBufferContext] = {}
        grad_buffer_ctx2module_prefix: dict[YaFSDPBufferContext] = {}
        state2modules_to_run_forward: dict[YaFSDPState, set[nn.Module]] = {}

        def validate_buffers(module: nn.Module, prefix: str = ""):
            if (state := _get_module_fsdp_state(module)) is not None and (
                param_group := state._fsdp_param_group
            ) is not None:
                if state not in state2modules_to_run_forward:
                    state2modules_to_run_forward[state] = set(state._modules)
                    if (
                        data_buffer_ctx := param_group._data_buffer_ctx
                    ) in data_buffer_ctx2module_prefix:
                        raise RuntimeError(
                            f"Param groups of {prefix} and {data_buffer_ctx2module_prefix[data_buffer_ctx]} share a"
                            " data buffer, so they can't be unsharded at the same time."
                        )
                    data_buffer_ctx2module_prefix[data_buffer_ctx] = prefix
                    if grad_buffer_ctx := param_group._grad_buffer_ctx is not None:
                        if (
                            grad_buffer_ctx := param_group._grad_buffer_ctx
                        ) in grad_buffer_ctx2module_prefix:
                            raise RuntimeError(
                                f"Param groups of {prefix} and {grad_buffer_ctx2module_prefix[grad_buffer_ctx]} share a"
                                " grad buffer, so they can't be unsharded at the same time."
                            )
                        grad_buffer_ctx2module_prefix[grad_buffer_ctx] = prefix
            for submodule_name, submodule in module.named_children():
                validate_buffers(
                    submodule, prefix + ("." if prefix else "") + submodule_name
                )
            if (state := _get_module_fsdp_state(module)) is not None and (
                param_group := state._fsdp_param_group
            ) is not None:
                modules_to_run_forward = state2modules_to_run_forward[state]
                modules_to_run_forward.remove(module)
                if len(modules_to_run_forward) == 0:
                    del data_buffer_ctx2module_prefix[param_group._data_buffer_ctx]
                    if (grad_buffer_ctx := param_group._grad_buffer_ctx) is not None:
                        del grad_buffer_ctx2module_prefix[grad_buffer_ctx]

        root_module = self._modules[0]
        validate_buffers(root_module)

    def _init_fqns(self) -> None:
        assert self._is_root
        root_module = self._modules[0]
        param_to_fsdp_param: dict[nn.Parameter, YaFSDPParam] = {}
        module_to_fsdp_param_group: dict[nn.Module, YaFSDPParamGroup] = {}
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
        self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        # When composing with module-hook-based activation checkpointing, the
        # the pre-backward hook is responsible for the unshard
        if self._training_state == TrainingState.PRE_BACKWARD:
            return args, kwargs
        self._training_state = TrainingState.FORWARD
        args, kwargs = self._root_pre_forward(module, args, kwargs)
        if self._mp_policy.cast_forward_inputs and self._mp_policy.param_dtype:
            with torch.profiler.record_function("YaFSDP::cast_forward_inputs"):
                cast_fn = functools.partial(
                    _cast_fp_tensor, self._mp_policy.param_dtype
                )
                args, kwargs = tree_map(cast_fn, args), tree_map(cast_fn, kwargs)
        if self._fsdp_param_group:
            args, kwargs = self._fsdp_param_group.pre_forward(module, args, kwargs)
        for fsdp_state in self._states_to_forward_prefetch:
            if (target_param_group := fsdp_state._fsdp_param_group) is not None:
                YaFSDPParamGroup._prefetch_unshard(target_param_group, "forward")
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
            default_prefetch = len(self._states_to_backward_prefetch) == 0
            self._fsdp_param_group.pre_backward(default_prefetch)
        for fsdp_state in self._states_to_backward_prefetch:
            if (target_param_group := fsdp_state._fsdp_param_group) is not None:
                YaFSDPParamGroup._prefetch_unshard(target_param_group, "backward")
        return grad

    def _root_post_backward_final_callback(self) -> None:
        logger.debug("YaFSDP::root_post_backward")
        with torch.profiler.record_function("YaFSDP::root_post_backward_callback"):
            for state in self._state_ctx.all_states:
                fsdp_param_group = state._fsdp_param_group
                if (
                    fsdp_param_group
                    and fsdp_param_group._training_state != TrainingState.POST_BACKWARD
                ):
                    # Run post-backward in case forward inputs did not require
                    # gradient so the autograd backward did not run
                    fsdp_param_group.post_backward()
                state._training_state = TrainingState.IDLE
                if fsdp_param_group:
                    fsdp_param_group._training_state = TrainingState.IDLE
                if self._state_ctx.is_last_backward:
                    state._finalize_backward()
                if fsdp_param_group is not None:
                    if (
                        yccl_handle := fsdp_param_group._data_buffer_ctx.yccl_handle
                    ) is not None:
                        yccl_handle.process_profiling_events()
                    if (
                        grad_buffer_ctx := fsdp_param_group._grad_buffer_ctx
                    ) is not None and (
                        yccl_handle := grad_buffer_ctx.yccl_handle
                    ) is not None:
                        yccl_handle.process_profiling_events()
            self._comm_ctx.post_forward_order.clear()
            self._state_ctx.post_backward_final_callback_queued = False

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
        Variable._execution_engine.queue_callback(
            self._root_post_backward_final_callback
        )


def _get_module_fsdp_state(module: nn.Module) -> YaFSDPState | None:
    state = _get_module_state(module)
    if isinstance(state, YaFSDPState):
        return state
    return None


def _register_group_forward_hooks(
    modules: Sequence[nn.Module],
    pre_hook: Callable,
    post_hook: Callable,
    modules_to_run: set[nn.Module],
):
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
        module.register_forward_pre_hook(
            wrapped_pre_hook, prepend=True, with_kwargs=True
        )
        for module in modules
    ]
    post_handles = [
        module.register_forward_hook(
            get_wrapped_post_hook(module), prepend=False, always_call=True
        )
        for module in modules
    ]
    return _MultiHandle(tuple(pre_handles + post_handles))


def get_buffer_ctx2ctx_using_param_groups_map(
    param_groups: list[YaFSDPParamGroup],
    buffer_ctx_attr_name: str,
    num_buffers: int | None,
    param_group_to_buffer_ctx_idx: dict[YaFSDPParamGroup, int] | None = None,
    buffer_ctx_idx_to_yccl_handle: dict[int, "yccl.Handle"] | None = None,
) -> dict[YaFSDPBufferContext, list[YaFSDPParamGroup]]:
    if param_group_to_buffer_ctx_idx is None:
        if num_buffers is None:
            raise ValueError(
                "num_buffers must be specified if param_group_to_buffer_ctx_idx is not specified."
            )
        param_group_to_buffer_ctx_idx = {
            param_group: index % num_buffers
            for index, param_group in enumerate(param_groups)
        }
    buffer_ctx2ctx_using_param_groups = {}
    if buffer_ctx_idx_to_yccl_handle is not None:
        buffer_ctx2yccl_handle = {}
    for param_group in param_groups:
        buffer_ctx_idx = param_group_to_buffer_ctx_idx[param_group]
        buffer_ctx2ctx_using_param_groups.setdefault(
            (buffer_ctx := getattr(param_groups[buffer_ctx_idx], buffer_ctx_attr_name)),
            [],
        ).append(param_group)
        if buffer_ctx_idx_to_yccl_handle is not None:
            buffer_ctx2yccl_handle[buffer_ctx] = buffer_ctx_idx_to_yccl_handle[
                buffer_ctx_idx
            ]
    return (
        buffer_ctx2ctx_using_param_groups,
        buffer_ctx2yccl_handle if buffer_ctx_idx_to_yccl_handle else None,
    )
