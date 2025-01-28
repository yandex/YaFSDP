from typing import List, Optional, Set, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._tensor import DeviceMesh, init_device_mesh
from torch.distributed.device_mesh import _get_device_handle

from ._common import FSDPMeshInfo, HSDPMeshInfo
from ._state import _get_module_fsdp_state


def _get_post_forward_mesh_info(
    reshard_after_forward: Union[bool, int], mesh_info: FSDPMeshInfo
) -> Optional[FSDPMeshInfo]:
    shard_mesh_size = mesh_info.shard_mesh_size
    if not isinstance(reshard_after_forward, (bool, int)):
        raise ValueError(
            "reshard_after_forward should be a bool or an int representing the "
            f"group size to reshard to, not {reshard_after_forward}"
        )
    # NOTE: `isinstance(False, int)` returns `True`.
    if not isinstance(reshard_after_forward, bool) and isinstance(reshard_after_forward, int):
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
            reshard_after_forward = False
        elif reshard_after_forward == shard_mesh_size:
            reshard_after_forward = True
    post_forward_mesh_info = None
    if reshard_after_forward is True:
        post_forward_mesh_info = mesh_info
    elif reshard_after_forward is not False:  # int case
        # For HSDP, we can flatten the two replicate dims into the 0th dim
        post_forward_mesh_tensor = mesh_info.mesh.mesh.view(-1, reshard_after_forward)
        post_forward_mesh = DeviceMesh(mesh_info.mesh.device_type, post_forward_mesh_tensor)
        post_forward_mesh_info = HSDPMeshInfo(post_forward_mesh, shard_mesh_dim=1, replicate_mesh_dim=0)
    return post_forward_mesh_info


def _init_default_fully_shard_mesh() -> DeviceMesh:
    """Default to global CUDA mesh if possible else global CPU mesh."""
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


def _get_managed_modules(root_modules: Tuple[nn.Module, ...]) -> List[nn.Module]:
    modules: List[nn.Module] = []
    root_modules_set = set(root_modules)
    # Track visited modules to avoid visiting shared modules multiple times
    visited_modules: Set[nn.Module] = set()

    def dfs(module: nn.Module) -> None:
        """
        Runs a DFS to collect managed modules, not recursing into modules with
        a non-composable API or ``fully_shard`` already applied.
        """
        if module not in root_modules_set and _get_module_fsdp_state(module) is not None:
            return  # nested `fully_shard` module
        visited_modules.add(module)
        for submodule in module.children():
            if submodule not in visited_modules:
                dfs(submodule)
        modules.append(module)

    for root_module in root_modules:
        dfs(root_module)
    return modules


def _verify_managed_param(name: str, param: nn.Parameter) -> None:
    """
    Verify if the parameter is accepted by fully_shard. The only restriction now
    is that the parameter cannot be a scalar tensor (param.numel == 0) since we
    need at least one dim to shard.
    """
    if len(param.shape) == 0:
        raise ValueError(
            "fully_shard doesn't support scalar parameters. " f"Change {name} to a 1D tensor with numel equal to 1."
        )


def _get_managed_states(
    modules: List[nn.Module],
) -> Tuple[List[nn.Parameter], List[torch.Tensor]]:
    params: List[nn.Parameter] = []
    buffers: List[torch.Tensor] = []
    # Track visited parameters/buffers to avoid visiting shared parameters and
    # buffers multiple times
    visited_params: Set[nn.Parameter] = set()
    visited_buffers: Set[torch.Tensor] = set()
    for module in modules:
        for name, param in module.named_parameters(recurse=False):
            if param not in visited_params:
                _verify_managed_param(name, param)
                params.append(param)
                visited_params.add(param)
        for buffer in module.buffers(recurse=False):
            if buffer not in visited_buffers:
                buffers.append(buffer)
                visited_buffers.add(buffer)
    return params, buffers


def _move_modules_to_device(
    modules: List[nn.Module],
    device: torch.device,
) -> None:
    for module in modules:
        module._apply(lambda t: torch.empty_like(t, device=device) if t.is_meta else t.to(device), recurse=False)


def _sync_states(
    params: List[nn.Parameter],
    buffers: List[torch.Tensor],
    process_group: dist.ProcessGroup,
    broadcast_bucket_size: int = 250 * 1024 * 1024,
    src: int = 0,
):
    states = [*(p.detach() for p in params), *buffers]
    if len(states) > 0:
        dist._broadcast_coalesced(process_group, states, broadcast_bucket_size, src)
