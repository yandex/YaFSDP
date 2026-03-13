from collections.abc import Iterable
from typing import (
    TYPE_CHECKING,
    Any,
    NoReturn,
    cast,
    overload,
)

import torch
import torch.nn as nn
from torch.distributed._composable import contract  # type: ignore[attr-defined]
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.utils import _get_root_modules

from ._api import (
    MixedPrecisionPolicy,
)
from ._common import FSDPMeshInfo
from ._init import (
    _get_device_from_mesh,
    _get_managed_modules,
    _get_managed_states,
    _get_post_forward_mesh_info,
    _init_default_fully_shard_mesh,
    _move_states_to_device,
)
from ._param_group import YaFSDPParamGroup
from ._state import YaFSDPState, _get_module_fsdp_state

if TYPE_CHECKING:
    from ._collectives import AllGatherResult

__all__ = [
    "UnshardHandle",
    "YaFSDPModule",
    "fully_shard",
]

cls_to_fsdp_cls: dict[type, type] = {}


@overload
def fully_shard(
    module: nn.Module,
    *,
    mesh: DeviceMesh | None = ...,
    reshard_after_forward: bool | None = ...,
    mp_policy: MixedPrecisionPolicy = ...,
    ignored_params: set[nn.Parameter] | None = ...,
    shard_alignment: int = ...,
) -> "YaFSDPModule": ...


@overload
def fully_shard(
    module: list[nn.Module],
    *,
    mesh: DeviceMesh | None = ...,
    reshard_after_forward: bool | None = ...,
    mp_policy: MixedPrecisionPolicy = ...,
    ignored_params: set[nn.Parameter] | None = ...,
    shard_alignment: int = ...,
) -> list["YaFSDPModule"]: ...


@contract(state_cls=YaFSDPState)  # type: ignore[misc]
def fully_shard(
    module: nn.Module | list[nn.Module],
    *,
    mesh: DeviceMesh | None = None,
    reshard_after_forward: bool | None = True,
    mp_policy: MixedPrecisionPolicy = MixedPrecisionPolicy(),  # noqa: B008
    ignored_params: set[nn.Parameter] | None = None,
    shard_alignment: int = 8,
) -> "YaFSDPModule | list[YaFSDPModule]":
    if isinstance(module, nn.ModuleList | nn.ModuleDict):
        raise ValueError(
            f"fully_shard does not support containers that do not implement forward: {module}"
        )
    mesh = mesh or _init_default_fully_shard_mesh()
    if mesh.ndim != 1:
        raise ValueError(f"fully_shard expects a 1D DeviceMesh but got {mesh}")
    else:
        mesh_info = FSDPMeshInfo(mesh, shard_mesh_dim=0)
    device = _get_device_from_mesh(mesh)
    # If the user does not provide ``reshard_after_forward``, we set it to True.
    # During lazy_init, we identify which module is the root and override its value to False
    post_forward_mesh_info = _get_post_forward_mesh_info(
        (
            reshard_after_forward
            if (auto_reshard_after_forward := reshard_after_forward is not None)
            else True
        ),
        mesh_info,
    )

    arg_module = module
    modules = (
        (module,) if isinstance(module, nn.Module) else tuple(_get_root_modules(module))
    )
    state = fully_shard.state(modules[0])  # type: ignore[attr-defined]
    state.init(modules, device, mp_policy, auto_reshard_after_forward)

    managed_modules = _get_managed_modules(modules, ignored_params)
    params, buffers = _get_managed_states(managed_modules, ignored_params)

    _move_states_to_device(params, buffers, device)
    if params:
        state._fsdp_param_group = YaFSDPParamGroup(
            params,
            modules,
            mesh_info,
            post_forward_mesh_info,
            device,
            mp_policy,
            shard_alignment,
        )

    # Place YaFSDP leftmost for highest priority in the method resolution order
    for module in modules:  # noqa: PLR1704
        cls = module.__class__
        new_cls = cls_to_fsdp_cls.get(cls, None)
        if not new_cls:
            dct = {"__deepcopy__": _unimplemented_deepcopy}
            new_cls = type(f"YaFSDP{cls.__name__}", (YaFSDPModule, cls), dct)
            cls_to_fsdp_cls[cls] = new_cls
        module.__class__ = new_cls
    return cast("YaFSDPModule", arg_module)


def _unimplemented_deepcopy(*args: Any, **kwargs: Any) -> NoReturn:
    raise AssertionError(
        "YaFSDP does not support deepcopy. Please use state dict for serialization."
    )


class YaFSDPModule:
    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        # Use index 2 since 0 is the dynamically constructed `YaFSDP<...>` class
        # and index 1 is the `YaFSDPModule` class itself
        orig_cls = cls.__mro__[2]
        self = orig_cls.__new__(orig_cls, *args, **kwargs)
        self.__init__(*args, **kwargs)
        return self

    def reshard(self) -> None:
        state = self._get_fsdp_state()
        if fsdp_param_group := state._fsdp_param_group:
            fsdp_param_group.reshard()

    def unshard(self, async_op: bool = False) -> "UnshardHandle | None":
        state = self._get_fsdp_state()
        fsdp_param_group = state._fsdp_param_group
        if fsdp_param_group is not None:
            fsdp_param_group.unshard(async_op=async_op)
        handle = _UnshardHandleImpl(fsdp_param_group)
        if async_op:
            return handle
        handle.wait()
        return None

    def set_is_last_backward(self, is_last_backward: bool) -> None:
        state = self._get_fsdp_state()
        state._state_ctx.is_last_backward = is_last_backward

    def set_requires_gradient_sync(
        self, requires_gradient_sync: bool, *, recurse: bool = True
    ) -> None:
        self_module = cast("nn.Module", self)
        modules = list(self_module.modules()) if recurse else [self_module]
        for module in modules:
            if isinstance(module, YaFSDPModule):
                state = module._get_fsdp_state()
                if fsdp_param_group := state._fsdp_param_group:
                    fsdp_param_group.reduce_grads = requires_gradient_sync
                    state.validate_shared_state()

    def set_reshard_after_forward(
        self, reshard_after_forward: bool, recurse: bool = True
    ) -> None:
        if not isinstance(reshard_after_forward, bool):
            raise ValueError(
                f"reshard_after_forward should be a bool, got {type(reshard_after_forward)}"
            )
        self_module = cast("nn.Module", self)
        modules = list(self_module.modules()) if recurse else [self_module]
        for module in modules:
            if isinstance(module, YaFSDPModule):
                state = module._get_fsdp_state()
                state._auto_reshard_after_forward = False
                if fsdp_param_group := state._fsdp_param_group:
                    fsdp_param_group.post_forward_mesh_info = (
                        _get_post_forward_mesh_info(
                            reshard_after_forward, fsdp_param_group.mesh_info
                        )
                    )

    def set_reshard_after_backward(
        self, reshard_after_backward: bool, *, recurse: bool = True
    ) -> None:
        self_module = cast("nn.Module", self)
        modules = list(self_module.modules()) if recurse else [self_module]
        for module in modules:
            if isinstance(module, YaFSDPModule):
                state = module._get_fsdp_state()
                if fsdp_param_group := state._fsdp_param_group:
                    fsdp_param_group.reshard_after_backward = reshard_after_backward
                    state.validate_shared_state()

    def set_modules_to_forward_prefetch(self, modules: "list[YaFSDPModule]") -> None:
        _assert_all_fsdp_modules(modules)
        self._get_fsdp_state()._states_to_forward_prefetch = [
            module._get_fsdp_state() for module in modules
        ]

    def set_modules_to_backward_prefetch(self, modules: "list[YaFSDPModule]") -> None:
        _assert_all_fsdp_modules(modules)
        self._get_fsdp_state()._states_to_backward_prefetch = [
            module._get_fsdp_state() for module in modules
        ]

    def set_post_optim_event(self, event: torch.Event) -> None:
        self._get_fsdp_state()._state_ctx.post_optim_event = event

    def set_unshard_in_backward(self, unshard_in_backward: bool) -> None:
        state = self._get_fsdp_state()
        if (fsdp_param_group := state._fsdp_param_group) is not None:
            fsdp_param_group.unshard_in_backward = unshard_in_backward

    def set_gradient_divide_factor(self, factor: float) -> None:
        state = self._get_fsdp_state()
        if (fsdp_param_group := state._fsdp_param_group) is not None:
            fsdp_param_group.gradient_divide_factor = factor

    def set_force_sum_reduction_for_comms(self, enable: bool) -> None:
        state = self._get_fsdp_state()
        if (fsdp_param_group := state._fsdp_param_group) is not None:
            fsdp_param_group.force_sum_reduction_for_comms = enable

    def _get_fsdp_state(self) -> YaFSDPState:
        if (state := _get_module_fsdp_state(cast("nn.Module", self))) is None:
            raise AssertionError(f"No YaFSDP state found on {self}")
        return state

    def _apply(self, *args: Any, **kwargs: Any) -> Any:
        # Reshard to ensure that sharded parameters are registered
        self.reshard()
        ret = super()._apply(*args, **kwargs)  # type: ignore[misc]
        state = self._get_fsdp_state()
        if not (fsdp_param_group := state._fsdp_param_group):
            return ret
        with torch.no_grad():
            for fsdp_param in fsdp_param_group.fsdp_params:
                fsdp_param.reset_sharded_param()
        return ret


class UnshardHandle:
    def wait(self) -> None:
        return


class _UnshardHandleImpl(UnshardHandle):
    def __init__(self, fsdp_param_group: YaFSDPParamGroup | None):
        self._fsdp_param_group = fsdp_param_group
        self._all_gather_result: AllGatherResult | None = None

    def wait(self) -> None:
        if self._fsdp_param_group is not None:
            self._all_gather_result = self._fsdp_param_group.wait_for_unshard()
            # Avoid keeping a reference
            self._fsdp_param_group = None
        elif (all_gather_result := self._all_gather_result) is not None:
            all_gather_result.wait()


def _assert_all_fsdp_modules(modules: Iterable[Any]) -> None:
    for module in modules:
        if not isinstance(module, YaFSDPModule):
            raise ValueError(f"Expects YaFSDPModule but got {type(module)}: {module}")
