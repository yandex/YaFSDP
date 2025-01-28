from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import torch


@dataclass(frozen=True)
class MixedPrecisionPolicy:
    param_dtype: Optional[torch.dtype] = None
    reduce_dtype: Optional[torch.dtype] = None
    output_dtype: Optional[torch.dtype] = None
    cast_forward_inputs: bool = True
    bit32_acc_for_bit16_reduce_scatter: bool = False

    def __post_init__(self):
        # Clamp `reduce_dtype` to `None` if no casting is required: since
        # gradients are computed in `param_dtype`, if `reduce_dtype` matches,
        # then we do not need extra casting
        if self.param_dtype == self.reduce_dtype:
            # Bypass the frozen dataclass checks
            object.__setattr__(self, "reduce_dtype", None)


class StateDictType(Enum):
    FULL_STATE_DICT = auto()
    SHARDED_STATE_DICT = auto()


@dataclass
class StateDictConfig:
    offload_to_cpu: bool = False


@dataclass
class FullStateDictConfig(StateDictConfig):
    rank0_only: bool = False


@dataclass
class ShardedStateDictConfig(StateDictConfig):
    pass


@dataclass
class StateDictSettings:
    state_dict_type: StateDictType
    state_dict_config: StateDictConfig
