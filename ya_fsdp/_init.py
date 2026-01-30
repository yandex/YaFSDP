import itertools
import logging

import torch
import torch.distributed as dist
import torch.nn as nn
from torch._logging import warning_once
from torch.distributed.device_mesh import (
    DeviceMesh,
    _get_device_handle,
    init_device_mesh,
)
from torch.distributed.tensor import DTensor
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

from ._common import FSDPMeshInfo
from ._param import ParamModuleInfo, unsafe_setattr_param
from ._state import _get_module_fsdp_state

logger = logging.getLogger("ya_fsdp")


def _get_post_forward_mesh_info(
    reshard_after_forward: bool | int, mesh_info: FSDPMeshInfo
) -> FSDPMeshInfo | None:
    shard_mesh_size = mesh_info.shard_mesh_size
    if not isinstance(reshard_after_forward, bool | int):
        raise ValueError(
            "reshard_after_forward should be a bool or an int representing the "
            f"group size to reshard to, not {reshard_after_forward}"
        )
    # NOTE: `isinstance(False, int)` returns `True`.
    if not isinstance(reshard_after_forward, bool) and isinstance(
        reshard_after_forward, int
    ):
        if (
            reshard_after_forward < 1
            or reshard_after_forward > shard_mesh_size
            or shard_mesh_size % reshard_after_forward != 0
        ):
            raise ValueError(
                "If passing reshard_after_forward as an int, it should be a "
                f"factor of {shard_mesh_size}, not {reshard_after_forward}"
            )
        elif reshard_after_forward == 1:
            msg = (
                "reshard_after_forward=1 (int) means resharding parameters to world size 1, "
                "instead of reshard_after_forward=True (bool)"
            )
            warning_once(logger, msg, stacklevel=2)
            reshard_after_forward = False
        elif reshard_after_forward == shard_mesh_size:
            reshard_after_forward = True
    post_forward_mesh_info = None
    if reshard_after_forward is True:
        post_forward_mesh_info = mesh_info
    elif reshard_after_forward is not False:  # int case
        raise ValueError(
            "YaFSDP does not support post forward resharding to a smaller world size yet."
        )
    return post_forward_mesh_info


def _init_default_fully_shard_mesh() -> DeviceMesh:
    if not dist.distributed_c10d.is_initialized():
        dist.distributed_c10d.init_process_group()
    default_pg = dist.distributed_c10d._get_default_group()
    device = torch._C._get_accelerator()
    mesh = init_device_mesh(device.type, mesh_shape=(default_pg.size(),))
    return mesh


def _get_device_from_mesh(mesh: DeviceMesh) -> torch.device:
    if mesh.device_type == "cpu":
        return torch.device("cpu")
    device_handle = _get_device_handle(mesh.device_type)
    return torch.device(mesh.device_type, device_handle.current_device())


def _ignore_module(
    module: nn.Module,
    ignored_params: set[nn.Parameter],
    ignore_decision: dict[nn.Module, bool],
) -> bool:
    if module in ignore_decision:
        return ignore_decision[module]

    if len(list(module.buffers(recurse=False))) > 0:
        # Cannot ignore a module with any buffer
        ignore_decision[module] = False
        return False

    for _, param in module.named_parameters(recurse=False):
        if param not in ignored_params:
            # at least one param is not ignored. So this module shouldn't be.
            ignore_decision[module] = False
            return False

    # Need to consider descendants of module
    for child in list(module.children()):
        ignore_child = _ignore_module(child, ignored_params, ignore_decision)
        if not ignore_child:
            # Cannot ignore module if one of its children is not ignored
            ignore_decision[module] = False
            return False

    # Safe to ignore module
    ignore_decision[module] = True
    return True


def _adjust_managed_modules(
    modules: list[nn.Module], ignored_params: set[nn.Parameter]
) -> list[nn.Module]:
    ignore_decision: dict[nn.Module, bool] = {}
    new_modules = []
    for module in modules:
        ignored = _ignore_module(module, ignored_params, ignore_decision)
        if not ignored:
            new_modules.append(module)
    return new_modules


def _get_managed_modules(
    root_modules: tuple[nn.Module, ...],
    ignored_params: set[nn.Parameter] | None = None,
) -> list[nn.Module]:
    modules: list[nn.Module] = []
    root_modules_set = set(root_modules)
    # Track visited modules to avoid visiting shared modules multiple times
    visited_modules: set[nn.Module] = set()

    def dfs(module: nn.Module) -> None:
        if (
            module not in root_modules_set
            and _get_module_fsdp_state(module) is not None
        ):
            return  # nested `fully_shard` module
        visited_modules.add(module)
        for submodule in module.children():
            if submodule not in visited_modules:
                dfs(submodule)
        modules.append(module)

    for root_module in root_modules:
        dfs(root_module)

    if ignored_params is None:
        return modules

    adjusted_modules = _adjust_managed_modules(modules, ignored_params)
    return adjusted_modules


def _verify_managed_param(name: str, param: nn.Parameter) -> None:
    if len(param.shape) == 0:
        raise ValueError(
            "fully_shard doesn't support scalar parameters. "
            f"Change {name} to a 1D tensor with numel equal to 1."
        )


def _get_managed_states(
    modules: list[nn.Module], ignored_params: set[nn.Parameter] | None = None
) -> tuple[list[nn.Parameter], list[torch.Tensor]]:
    params: list[nn.Parameter] = []
    buffers: list[torch.Tensor] = []
    # Track visited parameters/buffers to avoid visiting shared parameters and
    # buffers multiple times
    visited_params: set[nn.Parameter] = set()
    visited_buffers: set[torch.Tensor] = set()
    if ignored_params is None:
        ignored_params = set()

    for module in modules:
        for name, param in module.named_parameters(recurse=False):
            if param in ignored_params:
                # do not include an ignored parameters
                continue
            if param not in visited_params:
                _verify_managed_param(name, param)
                params.append(param)
                visited_params.add(param)
        for buffer in module.buffers(recurse=False):
            if buffer not in visited_buffers:
                buffers.append(buffer)
                visited_buffers.add(buffer)
    return params, buffers


def _move_states_to_device(
    params: list[nn.Parameter],
    buffers: list[torch.Tensor],
    param_module_infos: list[ParamModuleInfo],
    device: torch.device,
    param_dtype: torch.dtype | None = None,
) -> None:
    # Follow the logic in `nn.Module._apply`
    for tensor, module_info in itertools.chain(
        zip(params, param_module_infos, strict=False),
        zip(buffers, (None for _ in range(len(buffers))), strict=False),
    ):
        if tensor.device == device:
            continue
        if isinstance(tensor, DTensor):
            if (dtensor_mesh_type := tensor.device_mesh.device_type) != device.type:
                raise ValueError(
                    "Requires DTensor to have mesh of the same type as the FSDP mesh "
                    f"but got {dtensor_mesh_type} for DTensor and {device.type} for FSDP"
                )
            raise AssertionError(
                f"Expects DTensor to be moved to {dtensor_mesh_type} but got {tensor.device}"
            )
        if module_info is not None:
            if is_traceable_wrapper_subclass(tensor):
                inner_tensors, flatten_spec = tensor.__tensor_flatten__()
                inner_tensors = {
                    attr_name: torch.empty_like(
                        inner_tensor,
                        dtype=param_dtype if isinstance(tensor, nn.Parameter) else None,
                        device=device,
                    )
                    if (inner_tensor := getattr(tensor, attr_name)).is_meta
                    else inner_tensor.to(
                        dtype=param_dtype if isinstance(tensor, nn.Parameter) else None,
                        device=device,
                    )
                    for attr_name in inner_tensors
                }
                new_tensor = tensor.__tensor_unflatten__(
                    inner_tensors, flatten_spec, None, None
                )
            else:
                new_tensor = (
                    torch.empty_like(
                        tensor,
                        dtype=param_dtype if isinstance(tensor, nn.Parameter) else None,
                        device=device,
                    )
                    if tensor.is_meta
                    else tensor.to(
                        dtype=param_dtype if isinstance(tensor, nn.Parameter) else None,
                        device=device,
                    )
                )
            param = nn.Parameter(new_tensor, requires_grad=tensor.requires_grad)
            unsafe_setattr_param(module_info.module, module_info.param_name, param)
            for shared_module, shared_param_name in zip(
                module_info.shared_modules, module_info.shared_param_names, strict=False
            ):
                unsafe_setattr_param(shared_module, shared_param_name, param)
        else:
            tensor.data = tensor.to(
                dtype=param_dtype if isinstance(tensor, nn.Parameter) else None,
                device=device,
            )


def _sync_states(
    params: list[nn.Parameter],
    buffers: list[torch.Tensor],
    process_group: dist.ProcessGroup,
    broadcast_bucket_size: int = 250 * 1024 * 1024,
    src: int = 0,
):
    states = [*(p.detach() for p in params), *buffers]
    if len(states) > 0:
        dist._broadcast_coalesced(
            process_group,
            [
                state.to_local() if isinstance(state, DTensor) else state
                for state in states
            ],
            broadcast_bucket_size,
            src,
        )
        torch.cuda.empty_cache()
