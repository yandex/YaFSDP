from __future__ import annotations

import contextlib
from typing import (
    TYPE_CHECKING,
    Any,
    NoReturn,
    cast,
    overload,
)

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable import contract

from ._api import (
    FullStateDictConfig,
    MixedPrecisionPolicy,
    ShardedStateDictConfig,
    StateDictConfig,
    StateDictSettings,
    StateDictType,
)
from ._common import FSDPMeshInfo
from ._init import (
    _get_device_from_mesh,
    _get_managed_modules,
    _get_managed_states,
    _get_post_forward_mesh_info,
    _init_default_fully_shard_mesh,
    _move_states_to_device,
    _sync_states,
)
from ._param_group import (
    MultiDtypeYaFSDPParamGroup,
    YaFSDPParamGroup,
    _get_param_module_infos,
)
from ._state import YaFSDPState, _get_module_fsdp_state

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

    from torch.distributed._tensor import DeviceMesh

cls_to_fsdp_cls: dict[type, type] = {}


@overload
def fully_shard(
    module: nn.Module,
    *,
    mesh: DeviceMesh | None = ...,
    reshard_after_forward: bool | int = ...,
    mp_policy: MixedPrecisionPolicy = ...,
    ignored_params: set[nn.Parameter] | None = ...,
) -> YaFSDPModule: ...


@overload
def fully_shard(
    module: list[nn.Module],
    *,
    mesh: DeviceMesh | None = ...,
    reshard_after_forward: bool | int = ...,
    mp_policy: MixedPrecisionPolicy = ...,
    ignored_params: set[nn.Parameter] | None = ...,
) -> list[YaFSDPModule]: ...


# The decorator adds a state object to `module` that can be accessed via
# `fully_shard.state(module)`. The state object and module are 1:1.
@contract(state_cls=YaFSDPState)
def fully_shard(
    module: nn.Module | list[nn.Module],
    *,
    mesh: DeviceMesh | None = None,
    reshard_after_forward: bool = True,
    mp_policy: MixedPrecisionPolicy = MixedPrecisionPolicy(),  # noqa: B008
    ignored_params: set[nn.Parameter] | None = None,
    intra_node_pg: dist.ProcessGroup | None = None,
    orig_dtype: torch.dtype | None = None,
    shard_alignment: int = 8,
):
    if isinstance(module, nn.ModuleList | nn.ModuleDict):
        raise ValueError(
            f"fully_shard does not support containers that do not implement forward: {module}"
        )
    mesh = mesh or _init_default_fully_shard_mesh()
    if mesh.ndim != 1:
        raise ValueError(f"fully_shard expects a 1D DeviceMesh but got {mesh}")
    else:
        mesh_info = FSDPMeshInfo(
            mesh,
            shard_mesh_dim=0,
            intra_node_group=intra_node_pg or mesh.get_group(),
        )
    device = _get_device_from_mesh(mesh)
    post_forward_mesh_info = _get_post_forward_mesh_info(
        reshard_after_forward, mesh_info
    )

    arg_module = module
    modules = (module,) if isinstance(module, nn.Module) else tuple(module)
    state = fully_shard.state(modules[0])
    state.init(modules, device, mp_policy)

    managed_modules = _get_managed_modules(modules, ignored_params)
    params, buffers = _get_managed_states(managed_modules, ignored_params)

    param_module_infos = _get_param_module_infos(params, modules)
    _move_states_to_device(
        params, buffers, param_module_infos, device, param_dtype=orig_dtype
    )
    params, buffers = _get_managed_states(managed_modules, ignored_params)
    _sync_states(params, buffers, process_group=mesh_info.intra_node_group)
    if params:
        state._fsdp_param_group = (
            YaFSDPParamGroup(
                params,
                modules,
                mesh_info,
                post_forward_mesh_info,
                device,
                mp_policy,
                shard_alignment,
            )
            if mp_policy.all_gather_dtype_to_param_cls is None
            else MultiDtypeYaFSDPParamGroup(
                params,
                modules,
                mesh_info,
                post_forward_mesh_info,
                device,
                mp_policy,
                shard_alignment,
            )
        )
        del params
        torch.cuda.empty_cache()

    # Place YaFSDP leftmost for highest priority in the method resolution order
    for module in modules:  # noqa: PLR1704
        cls = module.__class__
        new_cls = cls_to_fsdp_cls.get(cls, None)
        if not new_cls:
            dct = {"__deepcopy__": unimplemented_deepcopy}
            new_cls = type(f"YaFSDP{cls.__name__}", (YaFSDPModule, cls), dct)
            cls_to_fsdp_cls[cls] = new_cls
            new_cls._version = 2
        module.__class__ = new_cls
    return arg_module


def unimplemented_deepcopy(*args: Any, **kwargs: Any) -> NoReturn:
    raise AssertionError(
        "YaFSDP does not support deepcopy. Please use state dict for serialization."
    )


class YaFSDPModule:
    def __new__(cls, *args, **kwargs):
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

    def unshard(self) -> None:
        state = self._get_fsdp_state()
        fsdp_param_group = state._fsdp_param_group
        if fsdp_param_group is not None:
            fsdp_param_group.lazy_init()
            fsdp_param_group.unshard()
            fsdp_param_group.wait_for_unshard()

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

                if (param_group := state._fsdp_param_group) is not None:
                    param_group.reduce_grads = requires_gradient_sync
                    state._validate_shared_state()

    def set_reshard_after_forward(
        self, reshard_after_forward: bool, recurse: bool = True
    ) -> None:
        self_module = cast("nn.Module", self)
        modules = list(self_module.modules()) if recurse else [self_module]
        for module in modules:
            if isinstance(module, YaFSDPModule):
                state = module._get_fsdp_state()
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
                    state._validate_shared_state()

    def set_modules_to_forward_prefetch(self, modules: list[YaFSDPModule]) -> None:
        _assert_all_fsdp_modules(modules)
        self._get_fsdp_state()._states_to_forward_prefetch = [
            module._get_fsdp_state() for module in modules
        ]

    def set_modules_to_backward_prefetch(self, modules: list[YaFSDPModule]) -> None:
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

    def set_state_dict_type(
        self,
        state_dict_type: StateDictType,
        state_dict_config: StateDictConfig | None = None,
    ) -> StateDictSettings:
        _state_dict_type_to_config: dict[StateDictType, type[StateDictConfig]] = {
            StateDictType.FULL_STATE_DICT: FullStateDictConfig,
            StateDictType.SHARDED_STATE_DICT: ShardedStateDictConfig,
        }

        # Use the default config if a state_dict config is not set.
        state_dict_config_type = _state_dict_type_to_config[state_dict_type]
        if state_dict_config is None:
            state_dict_config = state_dict_config_type()
        if state_dict_config_type is not type(state_dict_config):
            raise RuntimeError(
                f"Expected state_dict_config of type {state_dict_config_type} but got {type(state_dict_config)}"
            )

        # Set the state_dict type and configurations.
        param_groups = [
            param_group
            for state in set(self._get_fsdp_state()._state_ctx.all_states)
            if (param_group := state._fsdp_param_group) is not None
        ]
        prev_state_dict_types = {
            param_group._state_dict_type for param_group in param_groups
        }
        if len(prev_state_dict_types) != 1:
            raise AssertionError(
                f"YaFSDP expects uniform state_dict_type but got {prev_state_dict_types}"
            )
        prev_state_dict_type = next(iter(prev_state_dict_types))
        prev_state_dict_configs = []
        for param_group in param_groups:
            if (
                prev_state_dict_config := param_group._state_dict_config
            ) in prev_state_dict_configs:
                continue
            prev_state_dict_configs.append(prev_state_dict_config)
        if len(prev_state_dict_configs) != 1:
            raise AssertionError(
                f"YaFSDP expects uniform state_dict_config but got {prev_state_dict_configs}"
            )
        prev_state_dict_config = next(iter(prev_state_dict_configs))
        for param_group in param_groups:
            param_group._state_dict_type = state_dict_type
            param_group._state_dict_config = state_dict_config

        return StateDictSettings(prev_state_dict_type, prev_state_dict_config)

    @contextlib.contextmanager
    def state_dict_type(
        self,
        state_dict_type: StateDictType,
        state_dict_config: StateDictConfig | None = None,
    ) -> Generator:
        prev_state_dict_settings = self.set_state_dict_type(
            state_dict_type,
            state_dict_config,
        )
        yield
        self.set_state_dict_type(
            prev_state_dict_settings.state_dict_type,
            prev_state_dict_settings.state_dict_config,
        )


def _assert_all_fsdp_modules(modules: Iterable[Any]) -> None:
    for module in modules:
        if not isinstance(module, YaFSDPModule):
            raise ValueError(f"Expects YaFSDPModule but got {type(module)}: {module}")
