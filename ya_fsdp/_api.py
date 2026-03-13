from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MixedPrecisionPolicy:
    param_dtype: torch.dtype | None = None
    reduce_dtype: torch.dtype | None = None
    output_dtype: torch.dtype | None = None
    cast_forward_inputs: bool = False
    bit32_acc_for_bit16_reduce_scatter: bool = False
    all_gather_dtype_to_param_cls: dict[torch.dtype, type] | None = None

    def __post_init__(self) -> None:
        if self.bit32_acc_for_bit16_reduce_scatter and not (
            self.reduce_dtype == torch.bfloat16
        ):
            raise ValueError(
                "bit32_acc_for_bit16_reduce_scatter requires reduce dtype"
                f" to be bfloat16 , but got {self.reduce_dtype}."
            )
