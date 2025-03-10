import contextlib
from typing import Any, Dict, Generator, Iterable, List, NoReturn, Optional, Type, Union, cast

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable import contract
from torch.distributed._tensor import DeviceMesh

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
    _move_modules_to_device,
    _sync_states,
)
from ._param_group import YaFSDPParamGroup
from ._state import YaFSDPState, _get_module_fsdp_state

cls_to_fsdp_cls: Dict[Type, Type] = {}


# The decorator adds a state object to `module` that can be accessed via
# `fully_shard.state(module)`. The state object and module are 1:1.
@contract(state_cls=YaFSDPState)
def fully_shard(
    module: Union[nn.Module, List[nn.Module]],
    mesh: DeviceMesh,
    *,
    intra_node_pg: Optional[dist.ProcessGroup] = None,
    reshard_after_forward: bool = True,
    mp_policy: MixedPrecisionPolicy = MixedPrecisionPolicy(),
):
    """
    Apply YaFSDP to ``module``, where YaFSDP shards module parameters,
    gradients, and optimizer states across data parallel workers to save memory
    at the cost of communication.

    Args:
        module (Union[nn.Module, List[nn.Module]): The module or modules to
            shard with YaFSDP and group together for communication.
        # mesh (Optional[DeviceMesh]): This data parallel mesh defines the
        #     sharding and device. If 1D, then parameters are fully sharded
        #     across the 1D mesh (FSDP) with ``(Shard(0),)`` placement. If 2D,
        #     then parameters are sharded across the 1st dim and replicated
        #     across the 0th dim (HSDP) with ``(Replicate(), Shard(0))``
        #     placement. The mesh's device type gives the device type used for
        #     communication; if a CUDA or CUDA-like device type, then we use the
        #     current device.
        reshard_after_forward (bool): This controls the parameter
            behavior after forward and can trade off memory and communication:

            - If ``True``, then this reshards parameters after forward and
              re-all-gathers in backward.
            - If ``False``, then this keeps the unsharded parameters in memory
              after forward and avoids the all-gather in backward.
        mp_policy (MixedPrecisionPolicy): This controls the mixed precision
            policy, which offers parameter/reduction mixed precision for this
            module. See :class:`MixedPrecisionPolicy` for details.
    """
    if isinstance(module, (nn.ModuleList, nn.ModuleDict)):
        raise ValueError(f"fully_shard does not support containers that do not implement forward: {module}")
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
    post_forward_mesh_info = _get_post_forward_mesh_info(reshard_after_forward, mesh_info)

    arg_module = module
    modules = (module,) if isinstance(module, nn.Module) else tuple(module)
    state = fully_shard.state(modules[0])
    state.init(modules, device, mp_policy)

    managed_modules = _get_managed_modules(modules)
    _move_modules_to_device(managed_modules, device)
    params, buffers = _get_managed_states(managed_modules)
    _sync_states(params, buffers, process_group=mesh_info.intra_node_group)
    if params:
        state._fsdp_param_group = YaFSDPParamGroup(
            params,
            modules,
            mesh_info,
            post_forward_mesh_info,
            device,
            mp_policy,
        )
        del params
        torch.cuda.empty_cache()

    # Place YaFSDP leftmost for highest priority in the method resolution order
    for module in modules:
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
    raise AssertionError("YaFSDP does not support deepcopy. Please use state dict for serialization.")


class YaFSDPModule:
    def __new__(cls, *args, **kwargs):
        """
        Override ``__new__`` to remove the YaFSDP class and directly construct
        the original class for cases like indexing into a container module.
        """
        # Use index 2 since 0 is the dynamically constructed `YaFSDP<...>` class
        # and index 1 is the `YaFSDPModule` class itself
        orig_cls = cls.__mro__[2]
        self = orig_cls.__new__(orig_cls, *args, **kwargs)
        self.__init__(*args, **kwargs)
        return self

    def reshard(self) -> None:
        """
        Reshards the module's parameters, registering the sharded parameters
        to the module and freeing the unsharded parameters if needed. This
        method is *not* recursive.
        """
        state = self._get_fsdp_state()
        if fsdp_param_group := state._fsdp_param_group:
            fsdp_param_group.reshard()

    def unshard(self) -> None:
        """
        Unshards the module's parameters by allocating memory and all-gathering
        the parameters. This method is *not* recursive.
        """
        state = self._get_fsdp_state()
        fsdp_param_group = state._fsdp_param_group
        if fsdp_param_group is not None:
            fsdp_param_group.lazy_init()
            fsdp_param_group.unshard()
            fsdp_param_group.wait_for_unshard()

    def set_is_last_backward(self, is_last_backward: bool) -> None:
        """
        Sets whether the next backward is the last one, meaning that FSDP
        should wait for gradient reduction to finish and clear internal data
        structures used for explicit prefetching.
        """
        state = self._get_fsdp_state()
        state._state_ctx.is_last_backward = is_last_backward

    def set_requires_gradient_sync(self, requires_gradient_sync: bool, *, recurse: bool = True) -> None:
        """
        Sets if the module should sync gradients. This can be used to implement
        gradient accumulation without communication. For HSDP, this controls
        both reduce-scatter and all-reduce together.

        Args:
            requires_gradient_sync (bool): Whether to reduce gradients for the
                module's parameters.
            recurse (bool): Whether to set for all submodules or just the
                passed-in module.
        """
        self_module = cast(nn.Module, self)
        modules = list(self_module.modules()) if recurse else [self_module]
        for module in modules:
            if isinstance(module, YaFSDPModule):
                state = module._get_fsdp_state()

                if (param_group := state._fsdp_param_group) is not None:
                    param_group.reduce_grads = requires_gradient_sync
                    state._validate_shared_state()

    def set_reshard_after_backward(self, reshard_after_backward: bool, *, recurse: bool = True) -> None:
        """
        Sets if the module should reshard parameters after backward. This can
        be used during gradient accumulation to trade off higher memory for
        reduced communication.

        Args:
            reshard_after_backward (bool): Whether to reshard parameters after
                backward.
            recurse (bool): Whether to set for all submodules or just the
                passed-in module.
        """
        self_module = cast(nn.Module, self)
        modules = list(self_module.modules()) if recurse else [self_module]
        for module in modules:
            if isinstance(module, YaFSDPModule):
                state = module._get_fsdp_state()
                if fsdp_param_group := state._fsdp_param_group:
                    fsdp_param_group.reshard_after_backward = reshard_after_backward
                    state._validate_shared_state()

    def set_modules_to_forward_prefetch(self, modules: List["YaFSDPModule"]) -> None:
        """
        Sets the FSDP modules for which this FSDP module should explicitly
        prefetch all-gathers in forward. The prefetching runs after this
        module's all-gather copy-out.

        Passing a singleton list containing the next FSDP module gives the same
        all-gather overlap behavior as the default overlap behavior, except the
        prefetched all-gather is issued earlier from the CPU. Passing a list
        with at least length two is required for more aggressive overlap and
        will use more reserved memory.

        Args:
            modules (List[FSDPModule]): FSDP modules to prefetch.
        """
        _assert_all_fsdp_modules(modules)
        self._get_fsdp_state()._states_to_forward_prefetch = [module._get_fsdp_state() for module in modules]

    def set_modules_to_backward_prefetch(self, modules: List["YaFSDPModule"]) -> None:
        """
        Sets the FSDP modules for which this FSDP module should explicitly
        prefetch all-gathers in backward. This overrides the default backward
        pretching implementation that prefetches the next FSDP module based on
        the reverse post-forward order.

        Passing a singleton list containing the previous FSDP module gives the
        same all-gather overlap behavior as the default overlap behavior.
        Passing a list with at least length two is required for more aggressive
        overlap and will use more reserved memory.

        Args:
            modules (List[YaFSDPModule]): FSDP modules to prefetch.
        """
        _assert_all_fsdp_modules(modules)
        self._get_fsdp_state()._states_to_backward_prefetch = [module._get_fsdp_state() for module in modules]

    def set_post_optim_event(self, event: torch.Event) -> None:
        """
        Sets a post-optimizer-step event for the root FSDP module to wait the
        all-gather streams on.

        By default, the root FSDP module waits the all-gather streams on the
        current stream to ensure that the optimizer step has finished before
        all-gathering. However, this may introduce false dependencies if
        there is unrelated computation after the optimizer step. This API
        allows the user to provide their own event to wait on. After the root
        waits on the event, the event is discarded, so this API should be
        called with a new event each iteration.

        Args:
            event (torch.Event): Event recorded after the optimizer step
                to wait all-gather streams on.
        """
        self._get_fsdp_state()._state_ctx.post_optim_event = event

    def _get_fsdp_state(self) -> YaFSDPState:
        if (state := _get_module_fsdp_state(cast(nn.Module, self))) is None:
            raise AssertionError(f"No YaFSDP state found on {self}")
        return state

    def set_state_dict_type(
        self,
        state_dict_type: StateDictType,
        state_dict_config: Optional[StateDictConfig] = None,
    ) -> StateDictSettings:
        _state_dict_type_to_config: Dict[StateDictType, Type[StateDictConfig]] = {
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
        prev_state_dict_types = {param_group._state_dict_type for param_group in param_groups}
        if len(prev_state_dict_types) != 1:
            raise AssertionError(f"YaFSDP expects uniform state_dict_type but got {prev_state_dict_types}")
        prev_state_dict_type = next(iter(prev_state_dict_types))
        prev_state_dict_configs = []
        for param_group in param_groups:
            if (prev_state_dict_config := param_group._state_dict_config) in prev_state_dict_configs:
                continue
            prev_state_dict_configs.append(prev_state_dict_config)
        if len(prev_state_dict_configs) != 1:
            raise AssertionError(f"YaFSDP expects uniform state_dict_config but got {prev_state_dict_configs}")
        prev_state_dict_config = next(iter(prev_state_dict_configs))
        for param_group in param_groups:
            param_group._state_dict_type = state_dict_type
            param_group._state_dict_config = state_dict_config

        return StateDictSettings(prev_state_dict_type, prev_state_dict_config)

    @contextlib.contextmanager
    def state_dict_type(
        self,
        state_dict_type: StateDictType,
        state_dict_config: Optional[StateDictConfig] = None,
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
