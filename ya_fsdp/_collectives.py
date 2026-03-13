from typing import TYPE_CHECKING, Any, NamedTuple, Optional, cast

import torch
import torch.distributed as dist
from torch.distributed import ReduceOp
from torch.distributed.device_mesh import _get_device_handle

from ._param import YaFSDPParam

if TYPE_CHECKING:
    from ._param_group import YaFSDPParamGroup

    try:
        import yccl
    except ImportError:
        pass


class AllGatherResult(NamedTuple):
    all_gather_event: torch.Event | None
    all_gather_work: dist.distributed_c10d.Work | None

    def wait(self) -> None:
        if (event := self.all_gather_event) is not None:
            event.wait()
        if (work := self.all_gather_work) is not None:
            work.wait()


def all_gather(
    param_group: "YaFSDPParamGroup",
    fsdp_params: list[YaFSDPParam],
    padded_sharded_param_data: torch.Tensor,
    all_gather_input: torch.Tensor,
    all_gather_output: torch.Tensor,
    all_gather_group: dist.ProcessGroup,
    async_op: bool,
    all_gather_stream: torch.Stream,
    device: torch.device,
    param_dtype: torch.dtype | None,
    all_gather_dtype: torch.dtype | None,
    yccl_handle: Optional["yccl.Handle"],
) -> AllGatherResult:
    device_handle = _get_device_handle(device.type)
    with device_handle.stream(all_gather_stream):
        if not param_group.is_all_gather_input_set:
            for fsdp_param in fsdp_params:
                fsdp_param.set_all_gather_input()
            if all_gather_dtype is None and (
                param_dtype is not None or yccl_handle is not None
            ):
                with torch.autograd._unsafe_preserve_version_counter(all_gather_input):  # type: ignore[attr-defined]
                    all_gather_input.copy_(padded_sharded_param_data)
            param_group.is_all_gather_input_set = True
        if yccl_handle is None:
            all_gather_work = dist.all_gather_into_tensor(
                output_tensor=all_gather_output,
                input_tensor=all_gather_input,
                group=all_gather_group,
                async_op=async_op,
            )
        else:
            yccl_handle.all_gather(
                all_gather_input.view(torch.bfloat16),
                all_gather_output.view(torch.bfloat16),
            )
            all_gather_work = None
        all_gather_event = all_gather_stream.record_event()
    return AllGatherResult(
        all_gather_event,
        all_gather_work,
    )


def sizes_to_slices(sizes: list[int]) -> list[slice]:
    """Map a list of sizes to a list of slices.

    >>> sizes_to_slices([20, 20, 30])
    [slice(0, 20, None), slice(20, 40, None), slice(40, 70, None)]
    """
    offs = 0
    slices = []
    for sz in sizes:
        slices.append(slice(offs, offs + sz))
        offs += sz
    return slices


def reduce_scatter(
    param_group: "YaFSDPParamGroup",
    fsdp_params_with_grad: list[YaFSDPParam],
    padded_sharded_param_grad: torch.Tensor,
    padded_unsharded_param_grad: torch.Tensor,
    reduce_scatter_group: dist.ProcessGroup,
    reduce_scatter_stream: torch.Stream,
    orig_dtype: torch.dtype,
    param_dtype: torch.dtype | None,
    reduce_dtype: torch.dtype | None,
    device: Any,
    gradient_divide_factor: float | None,
    force_sum_reduction_for_comms: bool,
    bit32_acc_for_bit16_reduce_scatter: bool,
    yccl_handle: "yccl.Handle | None",
) -> torch.Event:
    device_handle = _get_device_handle(device.type)
    reduce_scatter_stream.wait_stream(device_handle.current_stream())
    with device_handle.stream(reduce_scatter_stream):
        reduce_in_sharded = (
            (
                (reduce_dtype is None and param_dtype is None)
                or reduce_dtype == orig_dtype
            )
            and all(
                fsdp_param.sharded_param.grad is None
                for fsdp_param in param_group.fsdp_params
                if fsdp_param.sharded_param.requires_grad
            )
            and yccl_handle is None
        )
        if reduce_in_sharded:
            output_tensor = padded_sharded_param_grad
        else:
            output_tensor = padded_unsharded_param_grad.chunk(
                reduce_scatter_group.size()
            )[reduce_scatter_group.rank()]
        predivide_factor, postdivide_factor, reduce_scatter_op = (
            _get_gradient_divide_factors(
                reduce_scatter_group,
                reduce_dtype or param_dtype or orig_dtype,
                gradient_divide_factor,
                force_sum_reduction_for_comms,
                yccl_handle=yccl_handle,
            )
        )
        _div_if_needed(padded_unsharded_param_grad, predivide_factor)
        if yccl_handle is None:
            dist.reduce_scatter_tensor(
                output_tensor,
                padded_unsharded_param_grad,
                op=reduce_scatter_op,
                group=reduce_scatter_group,
                **(
                    {"acc_type": torch.float32}
                    if bit32_acc_for_bit16_reduce_scatter
                    else {}
                ),
            )
        else:
            yccl_handle.reduce_scatter(padded_unsharded_param_grad)
        _div_if_needed(output_tensor, postdivide_factor)
        if not reduce_in_sharded:
            fsdp_params_which_require_grad = [
                fsdp_param
                for fsdp_param in param_group.fsdp_params
                if fsdp_param.sharded_param.requires_grad
            ]
            fsdp_params_without_sharded_grad = [
                fsdp_param
                for fsdp_param in fsdp_params_with_grad
                if fsdp_param.sharded_param.grad is None
            ]
            fsdp_params_with_sharded_grad = [
                fsdp_param
                for fsdp_param in fsdp_params_with_grad
                if fsdp_param.sharded_param.grad is not None
            ]
            if len(fsdp_params_without_sharded_grad) == len(
                fsdp_params_which_require_grad
            ):
                padded_sharded_param_grad.copy_(output_tensor)
            elif len(fsdp_params_with_sharded_grad) == len(
                fsdp_params_which_require_grad
            ):
                padded_sharded_param_grad.add_(output_tensor)
            else:
                torch._foreach_copy_(
                    [
                        cast("torch.Tensor", fsdp_param.sharded_param_grad)
                        for fsdp_param in fsdp_params_with_sharded_grad
                    ],
                    [
                        cast("torch.Tensor", fsdp_param.reduce_scatter_output)
                        for fsdp_param in fsdp_params_with_sharded_grad
                    ],
                )
                torch._foreach_add_(
                    [
                        cast("torch.Tensor", fsdp_param.sharded_param_grad)
                        for fsdp_param in fsdp_params_with_sharded_grad
                    ],
                    [
                        cast("torch.Tensor", fsdp_param.reduce_scatter_output)
                        for fsdp_param in fsdp_params_with_sharded_grad
                    ],
                )
        post_reduce_event = reduce_scatter_stream.record_event()
        for fsdp_param in fsdp_params_with_grad:
            fsdp_param.sharded_param.grad = fsdp_param.sharded_param_grad
        return post_reduce_event


def _get_gradient_divide_factors(
    reduce_scatter_group: dist.ProcessGroup,
    reduce_dtype: torch.dtype,
    factor: float | None,
    force_sum_reduction_for_comms: bool,
    yccl_handle: "yccl.Handle | None",
) -> tuple[float | None, float | None, dist.ReduceOp | dist.ReduceOp.RedOpType]:
    # YCCL only supports SUM reduction, hence we force it implicitly
    if yccl_handle is not None:
        force_sum_reduction_for_comms = True

    # For fp32/bf16, we do not need to worry about overflow/underflow, so we
    # use NCCL's built-in division to avoid separate div kernels
    overflow_risk = reduce_dtype not in (torch.float32, torch.bfloat16)

    data_parallel_size = reduce_scatter_group.size()

    if not overflow_risk and not force_sum_reduction_for_comms:
        if factor is None:
            # Warning: NCCL ReduceOp.AVG may produce incorrect results with
            # world size 1.
            if data_parallel_size == 1:
                return None, None, ReduceOp.SUM
            return None, None, ReduceOp.AVG
        if factor == reduce_scatter_group.size():
            reduce_scatter_op: dist.ReduceOp | dist.ReduceOp.RedOpType = ReduceOp.AVG
        else:
            reduce_scatter_op = torch.distributed._make_nccl_premul_sum(1 / factor)  # type: ignore[attr-defined]
        return None, None, reduce_scatter_op

    if factor is None:
        factor = float(data_parallel_size)
    pre_factor: float | None
    if overflow_risk:
        # Since fp16 has smaller dynamic range than fp32/bf16, we want to avoid
        # overflow/underflow. For N data parallel workers, each worker computes
        # g_i, and they collectively reduce (g_1 + ... + g_N) / N. To avoid
        # overflow/underflow, we divide by ~sqrt(N) before/after the reduction.
        pre_factor = 1
        while factor % pre_factor == 0 and factor / pre_factor > pre_factor:
            pre_factor *= 2
        post_factor = factor / pre_factor
    else:
        # Prefer post-multiplying as it operates on less data and is thus faster
        pre_factor, post_factor = None, factor

    return pre_factor, post_factor, ReduceOp.SUM


def _div_if_needed(tensor: torch.Tensor, div_factor: float | None) -> None:
    if div_factor is not None and div_factor > 1:
        tensor.div_(div_factor)
