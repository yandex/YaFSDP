from dataclasses import dataclass
from enum import Enum, auto

import torch


@dataclass(frozen=True)
class MixedPrecisionPolicy:
    param_dtype: torch.dtype | None = None
    reduce_dtype: torch.dtype | None = None
    output_dtype: torch.dtype | None = None
    cast_forward_inputs: bool = False
    bit32_acc_for_bit16_reduce_scatter: bool = False
    all_gather_dtype_to_param_cls: dict[torch.dtype, type] | None = None

    def __post_init__(self):
        if self.bit32_acc_for_bit16_reduce_scatter and not (
            self.param_dtype == self.reduce_dtype == torch.bfloat16
        ):
            raise ValueError(
                "bit32_acc_for_bit16_reduce_scatter can only be used with bfloat16 param and reduce dtypes"
                f", but got {self.param_dtype=} and {self.reduce_dtype=}."
            )


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
