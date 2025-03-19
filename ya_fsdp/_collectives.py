from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union, cast

import torch
import torch.distributed as dist
from torch.distributed import ReduceOp

from ._common import TrainingState
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
    padded_sharded_param_data_param_dtype: torch.Tensor,
    unsharded_param_data: torch.Tensor,
    data_buffer_ctx: "YaFSDPBufferContext",
    all_gather_group: dist.ProcessGroup,
    all_gather_stream: torch.Stream,
    param_dtype: Optional[torch.dtype],
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
        if param_dtype is not None:
            input_tensor = padded_sharded_param_data_param_dtype
            if param_group._training_state == TrainingState.FORWARD and not param_group.is_sharded_param_grad_set():
                input_tensor.copy_(padded_sharded_param_data)
        else:
            input_tensor = padded_sharded_param_data
        if yccl_handle is None:
            dist.all_gather_into_tensor(
                output_tensor=unsharded_param_data,
                input_tensor=input_tensor,
                group=all_gather_group,
            )
        else:
            yccl_handle.all_gather(input_tensor, unsharded_param_data)
    all_gather_event = all_gather_stream.record_event()
    return all_gather_event


def reduce_scatter(
    param_group: "YaFSDPParamGroup",
    fsdp_params_with_grad: List[YaFSDPParam],
    padded_sharded_param_grad: torch.Tensor,
    unsharded_param_grad: torch.Tensor,
    unsharded_param_grad_reduce_dtype: Optional[torch.Tensor],
    reduce_dtype_grad_buffer_ctx: Optional["YaFSDPBufferContext"],
    reduce_scatter_group: dist.ProcessGroup,
    reduce_scatter_stream: torch.Stream,
    orig_dtype: torch.dtype,
    param_dtype: Optional[torch.dtype],
    reduce_dtype: Optional[torch.dtype],
    device_handle: Any,
    bit32_acc_for_bit16_reduce_scatter: bool,
    yccl_handle: Optional["yccl.Handle"],
) -> Tuple[torch.Event, torch.Event]:
    reduce_scatter_stream.wait_stream(device_handle.current_stream())
    with device_handle.stream(reduce_scatter_stream):
        if reduce_dtype is not None:
            reduce_dtype_grad_buffer_ctx = cast("YaFSDPBufferContext", reduce_dtype_grad_buffer_ctx)
            unsharded_param_grad_reduce_dtype = cast(torch.Tensor, unsharded_param_grad_reduce_dtype)
            if (owner := reduce_dtype_grad_buffer_ctx.owner) is not None:
                raise RuntimeError(f"Reduce dtype gradient buffer already in use by {owner}")
            else:
                if (release_event := reduce_dtype_grad_buffer_ctx.release_event) is not None:
                    reduce_scatter_stream.wait_event(release_event)
                    reduce_dtype_grad_buffer_ctx.release_event = None
                reduce_dtype_grad_buffer_ctx.owner = param_group
            unsharded_param_grad_reduce_dtype.copy_(unsharded_param_grad)
            grad_buffer_release_event = reduce_scatter_stream.record_event()
            input_tensor = unsharded_param_grad_reduce_dtype
        else:
            input_tensor = unsharded_param_grad
        reduce_in_sharded = (
            (reduce_dtype is None and param_dtype is None or reduce_dtype == orig_dtype)
            and not param_group.is_sharded_param_grad_set()
            and yccl_handle is None
        )
        if reduce_in_sharded:
            output_tensor = padded_sharded_param_grad
        else:
            output_tensor = input_tensor.chunk(reduce_scatter_group.size())[reduce_scatter_group.rank()]
        predivide_factor, postdivide_factor = _get_gradient_divide_factors(
            reduce_scatter_group,
            reduce_dtype=reduce_dtype or param_dtype or orig_dtype,
            yccl_handle=yccl_handle,
        )
        _div_if_needed(input_tensor, predivide_factor)
        if yccl_handle is None:
            dist.reduce_scatter_tensor(
                output_tensor,
                input_tensor,
                op=ReduceOp.AVG if predivide_factor is None else ReduceOp.SUM,
                group=reduce_scatter_group,
                **{"acc_type": torch.float32} if bit32_acc_for_bit16_reduce_scatter else {},
            )
        else:
            yccl_handle.reduce_scatter(input_tensor)
        _div_if_needed(output_tensor, postdivide_factor)
        if not reduce_in_sharded:
            if param_group.is_sharded_param_grad_set():
                padded_sharded_param_grad.add_(output_tensor)
            else:
                padded_sharded_param_grad.copy_(output_tensor)
        post_reduce_event = reduce_scatter_stream.record_event()
        if reduce_dtype is not None:
            reduce_dtype_grad_buffer_ctx = cast("YaFSDPBufferContext", reduce_dtype_grad_buffer_ctx)
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
    yccl_handle: Optional["yccl.Handle"],
) -> Union[Tuple[None, None], Tuple[None, float], Tuple[float, float]]:
    # For fp32/bf16, we do not need to worry about overflow/underflow, so we
    # use NCCL's built-in division to avoid separate div kernels
    data_parallel_size = reduce_scatter_group.size()
    if reduce_dtype in (torch.float32, torch.bfloat16):
        if yccl_handle is None:
            return None, None
        else:
            return None, data_parallel_size
    # Since fp16 has smaller dynamic range than fp32/bf16, we want to avoid
    # overflow/underflow. For N data parallel workers, each worker computes
    # g_i, and they collectively reduce (g_1 + ... + g_N) / N. To avoid
    # overflow/underflow, we divide by ~sqrt(N) before/after the reduction.
    factor: int = 1
    while data_parallel_size % factor == 0 and data_parallel_size / factor > factor:
        factor *= 2
    return (float(factor), data_parallel_size / factor)


def _div_if_needed(tensor: torch.Tensor, div_factor: Optional[float]) -> None:
    if div_factor is not None and div_factor > 1:
        tensor.div_(div_factor)
