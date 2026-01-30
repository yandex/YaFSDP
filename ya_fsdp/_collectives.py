from typing import TYPE_CHECKING, Any, Optional, cast

import torch
import torch.distributed as dist
from torch.distributed import ReduceOp

from ._param import YaFSDPParam

if TYPE_CHECKING:
    from ._param_group import YaFSDPBufferContext, YaFSDPParamGroup

    try:
        import yccl
    except ImportError:
        pass


def all_gather(
    param_group: "YaFSDPParamGroup",
    padded_sharded_param_data: torch.Tensor,
    all_gather_input: torch.Tensor,
    all_gather_output: torch.Tensor,
    data_buffer_ctx: "YaFSDPBufferContext",
    all_gather_group: dist.ProcessGroup,
    all_gather_stream: torch.Stream,
    param_dtype: torch.dtype | None,
    all_gather_dtype: torch.dtype | None,
    device_handle: Any,
    yccl_handle: Optional["yccl.Handle"],
) -> torch.Event:
    if data_buffer_ctx.owner is not None:
        data_buffer_ctx.owner.reshard()
    if (release_event := data_buffer_ctx.release_event) is not None:
        all_gather_stream.wait_event(release_event)
        data_buffer_ctx.release_event = None
    data_buffer_ctx.owner = param_group
    with device_handle.stream(all_gather_stream):
        if not param_group._is_all_gather_input_set:
            for fsdp_param in param_group.fsdp_params:
                fsdp_param.set_all_gather_input()
            if all_gather_dtype is None and param_dtype is not None:
                all_gather_input.copy_(padded_sharded_param_data)
            param_group._is_all_gather_input_set = True
        if yccl_handle is None:
            dist.all_gather_into_tensor(
                output_tensor=all_gather_output,
                input_tensor=all_gather_input,
                group=all_gather_group,
            )
        else:
            yccl_handle.all_gather(
                all_gather_input.view(torch.bfloat16),
                all_gather_output.view(torch.bfloat16),
            )
    all_gather_event = all_gather_stream.record_event()
    return all_gather_event


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


def reduce_scatter(  # noqa: PLR0912
    param_group: "YaFSDPParamGroup",
    fsdp_params_with_grad: list[YaFSDPParam],
    padded_sharded_param_grad: torch.Tensor,
    unsharded_param_grad: torch.Tensor,
    unsharded_param_grad_reduce_dtype: torch.Tensor | None,
    reduce_dtype_grad_buffer_ctx: Optional["YaFSDPBufferContext"],
    reduce_scatter_group: dist.ProcessGroup,
    reduce_scatter_stream: torch.Stream,
    orig_dtype: torch.dtype,
    param_dtype: torch.dtype | None,
    reduce_dtype: torch.dtype | None,
    device_handle: Any,
    gradient_divide_factor: float | None,
    bit32_acc_for_bit16_reduce_scatter: bool,
    yccl_handle: Optional["yccl.Handle"],
    force_sum_reduction_for_comms: bool = False,
) -> tuple[torch.Event, torch.Event]:
    reduce_scatter_stream.wait_stream(device_handle.current_stream())
    with device_handle.stream(reduce_scatter_stream):
        if reduce_dtype is not None:
            reduce_dtype_grad_buffer_ctx = cast(
                "YaFSDPBufferContext", reduce_dtype_grad_buffer_ctx
            )
            unsharded_param_grad_reduce_dtype = cast(
                "torch.Tensor", unsharded_param_grad_reduce_dtype
            )
            if (owner := reduce_dtype_grad_buffer_ctx.owner) is not None:
                raise RuntimeError(
                    f"Reduce dtype gradient buffer already in use by {owner}"
                )
            else:
                if (
                    release_event := reduce_dtype_grad_buffer_ctx.release_event
                ) is not None:
                    reduce_scatter_stream.wait_event(release_event)
                    reduce_dtype_grad_buffer_ctx.release_event = None
                reduce_dtype_grad_buffer_ctx.owner = param_group
            unsharded_param_grad_reduce_dtype.copy_(unsharded_param_grad)
            grad_buffer_release_event = reduce_scatter_stream.record_event()
            input_tensor = unsharded_param_grad_reduce_dtype
        else:
            input_tensor = unsharded_param_grad
        reduce_in_sharded = (
            (
                (reduce_dtype is None and param_dtype is None)
                or reduce_dtype == orig_dtype
            )
            and not param_group.is_sharded_param_grad_set()
            and yccl_handle is None
        )
        if reduce_in_sharded:
            output_tensor = padded_sharded_param_grad
        else:
            output_tensor = input_tensor.chunk(reduce_scatter_group.size())[
                reduce_scatter_group.rank()
            ]
        predivide_factor, postdivide_factor, reduce_scatter_op = (
            _get_gradient_divide_factors(
                reduce_scatter_group,
                reduce_dtype=reduce_dtype or param_dtype or orig_dtype,
                factor=gradient_divide_factor,
                force_sum_reduction_for_comms=force_sum_reduction_for_comms,
                yccl_handle=yccl_handle,
            )
        )
        _div_if_needed(input_tensor, predivide_factor)
        if yccl_handle is None:
            dist.reduce_scatter_tensor(
                output_tensor,
                input_tensor,
                op=reduce_scatter_op,
                group=reduce_scatter_group,
                **(
                    {"acc_type": torch.float32}
                    if bit32_acc_for_bit16_reduce_scatter
                    else {}
                ),
            )
        else:
            yccl_handle.reduce_scatter(input_tensor)
        _div_if_needed(output_tensor, postdivide_factor)
        if not reduce_in_sharded:
            param_to_slice = dict(
                zip(
                    param_group.fsdp_params,
                    sizes_to_slices(param_group._sharded_param_numels),
                    strict=True,
                )
            )
            params_without_sharded_grad = list(
                filter(lambda p: p.sharded_param.grad is None, fsdp_params_with_grad)
            )
            if params_without_sharded_grad:
                torch._foreach_copy_(
                    [
                        padded_sharded_param_grad[param_to_slice[p]]
                        for p in params_without_sharded_grad
                    ],
                    [
                        output_tensor[param_to_slice[p]]
                        for p in params_without_sharded_grad
                    ],
                )
            params_with_sharded_grad = list(
                filter(
                    lambda p: p.sharded_param.grad is not None, fsdp_params_with_grad
                )
            )
            if params_with_sharded_grad:
                torch._foreach_add_(
                    [
                        padded_sharded_param_grad[param_to_slice[p]]
                        for p in params_with_sharded_grad
                    ],
                    [
                        output_tensor[param_to_slice[p]]
                        for p in params_with_sharded_grad
                    ],
                )
        post_reduce_event = reduce_scatter_stream.record_event()
        if reduce_dtype is not None:
            reduce_dtype_grad_buffer_ctx = cast(
                "YaFSDPBufferContext", reduce_dtype_grad_buffer_ctx
            )
            reduce_dtype_grad_buffer_ctx.release_event = post_reduce_event
            reduce_dtype_grad_buffer_ctx.owner = None
        else:
            grad_buffer_release_event = post_reduce_event
        for fsdp_param in fsdp_params_with_grad:
            fsdp_param.set_sharded_param_grad()
    return post_reduce_event, grad_buffer_release_event


def _get_gradient_divide_factors(
    reduce_scatter_group: dist.ProcessGroup,
    reduce_dtype: torch.dtype,
    factor: float | None = None,
    force_sum_reduction_for_comms: bool = False,
    yccl_handle: Optional["yccl.Handle"] = None,
) -> tuple[
    float | None,
    float | None,
    dist.ReduceOp | dist.ReduceOp.RedOpType,
]:
    # YCCL only supports SUM reduction, hence we force it implicitly
    if yccl_handle is not None:
        force_sum_reduction_for_comms = True

    # For fp32/bf16, we do not need to worry about overflow/underflow, so we
    # use NCCL's built-in division to avoid separate div kernels
    overflow_risk = reduce_dtype not in (torch.float32, torch.bfloat16)

    data_parallel_size = reduce_scatter_group.size()

    if factor is None:
        factor = float(data_parallel_size)

    if not overflow_risk and not force_sum_reduction_for_comms:
        if factor == data_parallel_size:
            # Warning: NCCL ReduceOp.AVG may produce incorrect results with
            # world size 1.
            if data_parallel_size == 1:
                return None, None, ReduceOp.SUM
            return None, None, ReduceOp.AVG
        else:
            reduce_scatter_op = torch.distributed._make_nccl_premul_sum(1 / factor)
            return None, None, reduce_scatter_op

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
