import traceback
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch.distributed.tensor import DeviceMesh


@dataclass
class DataParallelMeshInfo:
    mesh: DeviceMesh
    intra_node_group: dist.ProcessGroup
    shard_mesh_dim: Optional[int] = None
    replicate_mesh_dim: Optional[int] = None

    def __post_init__(self) -> None:
        if self.shard_mesh_dim is None and self.replicate_mesh_dim is None:
            raise AssertionError("At least one of shard_mesh_dim and replicate_mesh_dim must not be None")


@dataclass
class FSDPMeshInfo(DataParallelMeshInfo):
    def __post_init__(self) -> None:
        super().__post_init__()
        if self.shard_mesh_dim is None:
            raise AssertionError("Expects non-None shard_mesh_dim")
        self.shard_mesh_size = self.mesh.size(self.shard_mesh_dim)
        self.shard_process_group = self.mesh.get_group(self.shard_mesh_dim)
        self.shard_mesh_rank = self.shard_process_group.rank()


class TrainingState(Enum):
    """Describes the training state of one FSDP state / parameter group."""

    # Transition to forward starting pre-forward until post-forward
    FORWARD = auto()
    # Transition to pre-backward when unsharding in backward
    PRE_BACKWARD = auto()
    # Transition to post-backward when resharding and reducing gradients
    POST_BACKWARD = auto()
    # Idle before/after forward or before pre-backward/after post-backward
    IDLE = auto()


def _raise_assert_with_print(*args: Any, **kwargs: Any):
    print(f"[Rank {dist.get_rank()}] ", end="")
    print(*args, **kwargs)
    traceback.print_stack()
    raise AssertionError(*args, **kwargs)


def _cast_fp_tensor(dtype: torch.dtype, x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor) or not torch.is_floating_point(x) or x.dtype == dtype:
        return x
    return x.to(dtype)
