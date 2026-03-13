import contextlib
import logging
from collections.abc import Generator
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import _get_device_handle
from torch.distributed.fsdp._common_utils import _named_parameters_with_duplicates
from torch.profiler import record_function
from torch.utils._pytree import tree_flatten, tree_unflatten
from torch.utils.hooks import RemovableHandle

from ._api import MixedPrecisionPolicy
from ._collectives import AllGatherResult, all_gather, reduce_scatter
from ._common import FSDPMeshInfo, TrainingState, is_bw
from ._param import ParamModuleInfo, ShardedState, YaFSDPParam

if TYPE_CHECKING:
    try:
        import yccl
    except ImportError:
        pass

logger = logging.getLogger("ya_fsdp")

_ModuleToHandleDict = dict[nn.Module, RemovableHandle]  # for state dict


class YaFSDPBufferContext:
    class BufferType(Enum):
        ALL_GATHER = auto()
        REDUCE_SCATTER = auto()

    def __init__(self, buffer_type: BufferType):
        self._buffer_type = buffer_type

    def lazy_init(
        self,
        buffer_size_in_bytes: int,
        device: torch.device,
        yccl_handle: "yccl.Handle | None" = None,
    ) -> None:
        buffer_size_in_bfloat16 = buffer_size_in_bytes // torch.bfloat16.itemsize
        self.buffer = (
            torch.empty(buffer_size_in_bfloat16, dtype=torch.bfloat16, device=device)
            if yccl_handle is None
            else cast(
                "torch.Tensor",
                getattr(
                    yccl_handle,
                    {
                        self.BufferType.ALL_GATHER: "add_all_gather_output_buffer",
                        self.BufferType.REDUCE_SCATTER: "add_reduce_scatter_buffer",
                    }[self._buffer_type],
                )(buffer_size_in_bfloat16),
            )
        ).view(torch.uint8)
        self.owner: YaFSDPParamGroup | None = None
        self.release_event: torch.Event | None = None
        self.yccl_handle = yccl_handle


class YaFSDPCommContext:
    def lazy_init(self, device: torch.device) -> None:
        self.device_handle = _get_device_handle(device.type)
        # Setting the all-gather/reduce-scatter streams to be higher priority
        # can help avoid some issues where their copies in/out are delayed and
        # block computation (this is different from high-pri NCCL streams)
        high_priority = -1
        # All-gather stream allows overlapping next all-gather with current
        # forward compute
        self.all_gather_stream = cast(
            "torch.Stream", self.device_handle.Stream(priority=high_priority)
        )
        # Reduce-scatter stream gives separate execution "thread" for post-
        # backward logic like pre/post-gradient division and reduce-scatter
        self.reduce_scatter_stream = self.all_gather_stream
        # Post-forward order for explicit backward prefetching
        self.post_forward_order: list[YaFSDPParamGroup] = []  # will cause ref cycles

    def get_all_gather_stream(
        self, async_op: bool, training_state: TrainingState
    ) -> torch.Stream:
        if not async_op and training_state in (
            TrainingState.FORWARD,
            TrainingState.PRE_BACKWARD,
        ):
            return self.all_gather_stream
        current_stream = cast("torch.Stream", self.device_handle.current_stream())
        return current_stream


class YaFSDPParamGroup:
    orig_dtype: torch.dtype
    param_dtype: torch.dtype | None
    reduce_dtype: torch.dtype | None
    _all_gather_input: dict[torch.dtype | None, torch.Tensor]
    _all_gather_output: dict[torch.dtype | None, torch.Tensor]
    _reduce_scatter_input: dict[torch.dtype | None, torch.Tensor | None]
    _reduce_scatter_output: dict[torch.dtype | None, torch.Tensor | None]

    def __init__(  # noqa: PLR0915
        self,
        params: list[nn.Parameter],
        modules: tuple[nn.Module, ...],
        mesh_info: FSDPMeshInfo,
        post_forward_mesh_info: FSDPMeshInfo | None,
        device: torch.device,
        mp_policy: MixedPrecisionPolicy,
        shard_alignment: int,
    ):
        self.modules = modules  # permit ref cycle because 1:1 lifetime
        param_module_infos = _get_param_module_infos(params, modules)

        self.fsdp_params = [
            YaFSDPParam(
                param,
                module_info,
                mesh_info,
                post_forward_mesh_info,
                device,
            )
            for param, module_info in zip(params, param_module_infos, strict=True)
        ]
        self.mesh_info = mesh_info
        self.post_forward_mesh_info = post_forward_mesh_info
        self.device = device
        self.device_handle = _get_device_handle(device.type)
        self.mp_policy = mp_policy
        self._training_state = TrainingState.IDLE
        # Group's sharded state always matches its parameters' sharded states
        self._sharded_state = ShardedState.SHARDED
        self._module_fqn: str | None = None  # prefixed from root module

        # - Hook state
        self._module_to_pre_save_state_dict_hook_handle: _ModuleToHandleDict = {}
        self._module_to_pre_load_state_dict_hook_handle: _ModuleToHandleDict = {}

        # - Communication and communication/computation overlap
        self.comm_ctx = YaFSDPCommContext()
        # Group's indices in the shared post-forward order
        self._post_forward_indices: list[int] = []
        # Whether to reduce gradients at all (whether for FSDP or HSDP)
        self.reduce_grads: bool = True
        # Whether to reshard parameters after backward (only useful for
        # gradient accumulation)
        self.reshard_after_backward: bool = True
        # Optional custom factor for the gradient reduction op (e.g. to divide
        # by a factor other than the world size)
        self.gradient_divide_factor: float | None = None
        # Whether reduce-scatter and all-reduce should be issued using only
        # summations, potentially with separate pre-/post-scaling.
        self.force_sum_reduction_for_comms: bool = False
        # `async_op` arg used for pre-forward/pre-backward unshard; can be
        # overridden to only do explicit prefetching and avoid inter-stream
        # fragmentation from using separate unshard streams
        self.unshard_async_op: bool = False
        # Whether to unshard in backward: can be overridden by the user if the
        # parameters in this group are not needed for backward (e.g. embedding)
        self.unshard_in_backward: bool = True

        # - CUDA events for stream synchronization
        # Holds all-gather sync objects
        self._all_gather_result: AllGatherResult | None = None
        # Holds the CUDA event that marks the end of the group's post-backward,
        # which should be waited on at the end of backward
        self._post_reduce_event: torch.Event | None = None

        self._init_mp_dtypes()
        param_group_requires_grad = any(param.requires_grad for param in params)

        self.data_buffer_ctx = YaFSDPBufferContext(
            buffer_type=YaFSDPBufferContext.BufferType.ALL_GATHER
        )
        self.grad_buffer_ctx = (
            YaFSDPBufferContext(
                buffer_type=YaFSDPBufferContext.BufferType.REDUCE_SCATTER
            )
            if param_group_requires_grad
            else None
        )

        self._all_gather_dtype_to_fsdp_params: dict[
            torch.dtype | None, list[YaFSDPParam]
        ] = {
            **{
                None: [
                    fsdp_param
                    for fsdp_param in self.fsdp_params
                    if (
                        mp_policy.all_gather_dtype_to_param_cls is None
                        or not any(
                            isinstance(fsdp_param.param, param_cls)
                            for param_cls in mp_policy.all_gather_dtype_to_param_cls.values()
                        )
                    )
                ]
            },
            **(
                {
                    all_gather_dtype: [
                        fsdp_param
                        for fsdp_param in self.fsdp_params
                        if isinstance(fsdp_param.param, param_cls)
                    ]
                    for all_gather_dtype, param_cls in mp_policy.all_gather_dtype_to_param_cls.items()
                }
                if mp_policy.all_gather_dtype_to_param_cls is not None
                else {}
            ),
        }

        self._unsharded_param_numels: dict[torch.dtype | None, list[int]] = {}
        self.padded_unsharded_param_numel: dict[torch.dtype | None, int] = {}
        self._padded_sharded_param_numel: dict[torch.dtype | None, int] = {}
        self._padded_sharded_param_data: dict[torch.dtype | None, torch.Tensor] = {}
        self.is_all_gather_input_set = False
        self._padded_sharded_param_grad: dict[
            torch.dtype | None, torch.Tensor | None
        ] = {}
        self._sharded_param_numels: dict[torch.dtype | None, list[int]] = {}
        sharded_data_global_offsets: dict[torch.dtype | None, list[int]] = {}

        for (
            all_gather_dtype,
            fsdp_params,
        ) in self._all_gather_dtype_to_fsdp_params.items():
            shard_world_size = self.mesh_info.shard_mesh_size

            self._unsharded_param_numels[all_gather_dtype] = unsharded_param_numels = [
                fsdp_param.param_data.numel() for fsdp_param in fsdp_params
            ]

            padded_unsharded_param_numel = sum(unsharded_param_numels)
            divider = shard_world_size * shard_alignment
            if padded_unsharded_param_numel % divider != 0:
                padded_unsharded_param_numel += (
                    divider - padded_unsharded_param_numel % divider
                )
            self.padded_unsharded_param_numel[all_gather_dtype] = (
                padded_unsharded_param_numel
            )
            padded_sharded_param_numel = (
                padded_unsharded_param_numel // shard_world_size
            )
            self._padded_sharded_param_numel[all_gather_dtype] = (
                padded_sharded_param_numel
            )

            self._padded_sharded_param_data[all_gather_dtype] = (
                padded_sharded_param_data
            ) = torch.empty(
                padded_sharded_param_numel, dtype=self.orig_dtype, device=device
            )

            self._padded_sharded_param_grad[all_gather_dtype] = (
                padded_sharded_param_grad
            ) = (
                torch.zeros_like(padded_sharded_param_data)
                if param_group_requires_grad
                else None
            )

            padded_unsharded_param_data = torch.empty(
                padded_unsharded_param_numel, dtype=self.orig_dtype, device=device
            )
            max_param_indices_dtype_value = torch.iinfo(
                param_indices_dtype := torch.uint16
            ).max
            assert len(fsdp_params) < max_param_indices_dtype_value
            max_element_indices_dtype_value = torch.iinfo(
                element_indices_dtype := torch.int64
            ).max
            assert max(unsharded_param_numels) < max_element_indices_dtype_value
            padded_unsharded_param_indices = torch.full_like(
                padded_unsharded_param_data,
                fill_value=max_param_indices_dtype_value,
                dtype=param_indices_dtype,
            )
            padded_unsharded_param_element_indices = torch.full_like(
                padded_unsharded_param_data,
                fill_value=max_element_indices_dtype_value,
                dtype=element_indices_dtype,
            )
            for param_index, (
                fsdp_param,
                unsharded_param_numel,
                unsharded_param_data,
                unsharded_param_indices,
                unsharded_param_element_indices,
            ) in enumerate(
                zip(
                    self.fsdp_params,
                    unsharded_param_numels,
                    padded_unsharded_param_data[: sum(unsharded_param_numels)].split(
                        unsharded_param_numels
                    ),
                    padded_unsharded_param_indices[: sum(unsharded_param_numels)].split(
                        unsharded_param_numels
                    ),
                    padded_unsharded_param_element_indices[
                        : sum(unsharded_param_numels)
                    ].split(unsharded_param_numels),
                    strict=True,
                )
            ):
                with torch.no_grad():
                    unsharded_param_data.copy_(fsdp_param.param_data.view(-1))
                unsharded_param_indices.copy_(param_index)
                unsharded_param_element_indices.copy_(
                    torch.arange(
                        unsharded_param_numel,
                        dtype=element_indices_dtype,
                        device=device,
                    )
                )

            shard_rank = self.mesh_info.shard_mesh_rank

            padded_sharded_param_data.copy_(
                torch.chunk(padded_unsharded_param_data, shard_world_size)[shard_rank]
            )

            per_rank_padded_sharded_param_indices = torch.chunk(
                padded_unsharded_param_indices, shard_world_size
            )
            padded_sharded_param_indices = per_rank_padded_sharded_param_indices[
                shard_rank
            ]
            per_rank_padded_sharded_param_element_indices = torch.chunk(
                padded_unsharded_param_element_indices, shard_world_size
            )
            padded_sharded_param_element_indices = (
                per_rank_padded_sharded_param_element_indices[shard_rank]
            )

            per_rank_sharded_param_numels = [
                tuple(
                    cast("int", padded_sharded_param_indices.eq(index).sum().item())
                    for padded_sharded_param_indices in per_rank_padded_sharded_param_indices
                )
                for index, _ in enumerate(self.fsdp_params)
            ]
            self._sharded_param_numels[all_gather_dtype] = sharded_param_numels = [
                cast("int", padded_sharded_param_indices.eq(index).sum().item())
                for index, _ in enumerate(self.fsdp_params)
            ]
            sharded_data_global_offsets[all_gather_dtype] = [
                (
                    cast("int", element_indices.min().item())
                    if (
                        element_indices := padded_sharded_param_element_indices[
                            padded_sharded_param_indices.eq(index)
                        ]
                    ).numel()
                    > 0
                    else self._unsharded_param_numels[all_gather_dtype][index]
                )
                for index in range(len(self.fsdp_params))
            ]

            for (
                fsdp_param,
                param_sharded_param_data,
                param_sharded_param_grad,
                global_offset,
                shard_numels,
            ) in zip(
                fsdp_params,
                padded_sharded_param_data[: sum(sharded_param_numels)].split(
                    sharded_param_numels
                ),
                (
                    padded_sharded_param_grad[: sum(sharded_param_numels)].split(
                        sharded_param_numels
                    )
                    if padded_sharded_param_grad is not None
                    else (None,) * len(sharded_param_numels)
                ),
                sharded_data_global_offsets[all_gather_dtype],
                per_rank_sharded_param_numels,
                strict=True,
            ):
                fsdp_param.init_sharded_param(
                    param_sharded_param_data,
                    param_sharded_param_grad,
                    global_offset=global_offset,
                    shard_numels=shard_numels,
                )

    # Initialization #
    def _init_unsharded_params(self) -> None:
        self._all_gather_output = {}
        self._reduce_scatter_input = {}
        for (
            all_gather_dtype,
            fsdp_params,
        ) in self._all_gather_dtype_to_fsdp_params.items():
            padded_unsharded_param_numel = self.padded_unsharded_param_numel[
                all_gather_dtype
            ]
            self._all_gather_output[all_gather_dtype] = (
                self.data_buffer_ctx.buffer.view(
                    all_gather_dtype or self.param_dtype or self.orig_dtype
                ).narrow(0, 0, padded_unsharded_param_numel)
            )
            self._reduce_scatter_input[all_gather_dtype] = (
                self.grad_buffer_ctx.buffer.view(
                    self.reduce_dtype or self.param_dtype or self.orig_dtype
                ).narrow(0, 0, padded_unsharded_param_numel)
                if self.grad_buffer_ctx is not None
                else None
            )

            unsharded_param_numels = self._unsharded_param_numels[all_gather_dtype]
            all_gather_output = self._all_gather_output[all_gather_dtype]
            reduce_scatter_input = self._reduce_scatter_input[all_gather_dtype]
            for fsdp_param, param_all_gather_output, param_reduce_scatter_input in zip(
                fsdp_params,
                all_gather_output[: sum(unsharded_param_numels)].split(
                    unsharded_param_numels
                ),
                (
                    reduce_scatter_input[: sum(unsharded_param_numels)].split(
                        self._unsharded_param_numels[all_gather_dtype]
                    )
                    if reduce_scatter_input is not None
                    else (None,) * len(unsharded_param_numels)
                ),
                strict=True,
            ):
                fsdp_param.init_unsharded_param(
                    param_all_gather_output, param_reduce_scatter_input
                )

    def _init_mp_dtypes(self) -> None:
        for fsdp_param in self.fsdp_params:
            fsdp_param.init_dtype_attrs(self.mp_policy)
        trainable_params: list[YaFSDPParam] = [
            p for p in self.fsdp_params if p.param.requires_grad
        ]
        orig_dtypes = {p.orig_dtype for p in self.fsdp_params}
        param_dtypes = {p.param_dtype for p in self.fsdp_params}
        reduce_dtypes = {p.reduce_dtype for p in trainable_params}
        if len(trainable_params) > 0 and len(orig_dtypes) != 1:
            raise AssertionError(
                f"YaFSDP expects uniform original parameter dtype but got {orig_dtypes}"
            )
        self.orig_dtype = next(iter(orig_dtypes))
        if len(param_dtypes) != 1:
            raise AssertionError(
                f"YaFSDP expects uniform param dtype but got {param_dtypes}"
            )
        self.param_dtype = next(iter(param_dtypes))
        if len(trainable_params) > 0 and len(reduce_dtypes) != 1:
            # Models may have no grad params
            raise AssertionError(
                f"YaFSDP expects uniform reduce dtype but got {reduce_dtypes}"
            )
        self.reduce_dtype = (
            next(iter(reduce_dtypes)) if len(trainable_params) > 0 else None
        )

    def lazy_init(self) -> None:
        # Lazy init should be idempotent
        if not hasattr(self.comm_ctx, "device_handle"):
            self.comm_ctx.device_handle = _get_device_handle(self.device.type)
        self._validate_no_meta_params()
        self._register_state_dict_hooks()
        self._init_unsharded_params()
        self._all_gather_input = {}
        self._reduce_scatter_output = {}
        for (
            all_gather_dtype,
            fsdp_params,
        ) in self._all_gather_dtype_to_fsdp_params.items():
            if (yccl_handle := self.data_buffer_ctx.yccl_handle) is None:
                self._all_gather_input[all_gather_dtype] = (
                    self._padded_sharded_param_data[all_gather_dtype]
                    if all_gather_dtype is None and self.param_dtype is None
                    else torch.empty_like(
                        self._padded_sharded_param_data[all_gather_dtype],
                        dtype=all_gather_dtype or self.param_dtype,
                    )
                )
            else:
                self._all_gather_input[all_gather_dtype] = (
                    yccl_handle.add_all_gather_input_buffer(
                        self._padded_sharded_param_numel[all_gather_dtype]
                        * (
                            all_gather_dtype or self.param_dtype or self.orig_dtype
                        ).itemsize
                        // torch.bfloat16.itemsize
                    ).view(all_gather_dtype or self.param_dtype or self.orig_dtype)
                )
            for fsdp_param, param_all_gather_input in zip(
                fsdp_params,
                self._all_gather_input[all_gather_dtype][
                    : sum(self._sharded_param_numels[all_gather_dtype])
                ].split(self._sharded_param_numels[all_gather_dtype]),
                strict=True,
            ):
                fsdp_param.init_all_gather_input(param_all_gather_input)
            if self.grad_buffer_ctx is not None:
                reduce_scatter_input = self._reduce_scatter_input[all_gather_dtype]
                assert reduce_scatter_input is not None
                padded_sharded_param_grad = self._padded_sharded_param_grad[
                    all_gather_dtype
                ]
                assert padded_sharded_param_grad is not None
                self._reduce_scatter_output[all_gather_dtype] = (
                    reduce_scatter_output
                ) = reduce_scatter_input.chunk(
                    self._reduce_scatter_process_group.size()
                )[self._reduce_scatter_process_group.rank()]
                for fsdp_param, param_reduce_scatter_output in zip(
                    fsdp_params,
                    reduce_scatter_output[
                        : sum(self._sharded_param_numels[all_gather_dtype])
                    ].split(self._sharded_param_numels[all_gather_dtype]),
                    strict=True,
                ):
                    fsdp_param.init_reduce_scatter_output(param_reduce_scatter_output)

    # Runtime #
    def unshard(self, async_op: bool = False) -> None:
        if self._all_gather_result is not None:  # already called, pending wait
            return
        if self.is_unsharded:
            return  # no-op
        if (
            not self.unshard_in_backward
            and self._training_state == TrainingState.PRE_BACKWARD
        ):
            return
        logger.debug(
            "%s", self._with_fqn(f"YaFSDP::{self._training_state.name.lower()}_unshard")
        )
        with record_function(self._with_fqn("YaFSDP::all_gather")):
            all_gather_stream = self.comm_ctx.get_all_gather_stream(
                async_op, self._training_state
            )
            if (
                owner := (data_buffer_ctx := self.data_buffer_ctx).owner
            ) is not None and owner is not self:
                raise RuntimeError(
                    f"{self} tried to acquire its data buffer, but it is in use by {owner}."
                )
            if (release_event := data_buffer_ctx.release_event) is not None:
                all_gather_stream.wait_event(release_event)
                data_buffer_ctx.release_event = None
            data_buffer_ctx.owner = self
            for (
                all_gather_dtype,
                fsdp_params,
            ) in self._all_gather_dtype_to_fsdp_params.items():
                self._all_gather_result = all_gather(
                    self,
                    fsdp_params,
                    self._padded_sharded_param_data[all_gather_dtype],
                    self._all_gather_input[all_gather_dtype],
                    self._all_gather_output[all_gather_dtype],
                    self._all_gather_process_group,
                    async_op,
                    all_gather_stream,
                    self.device,
                    self.param_dtype,
                    all_gather_dtype,
                    self.data_buffer_ctx.yccl_handle,
                )

    def wait_for_unshard(self) -> AllGatherResult | None:
        if (all_gather_result := self._all_gather_result) is None:
            return None
        if all_gather_result is not None:
            all_gather_result.wait()
        self._to_unsharded()
        self._all_gather_result = None
        return all_gather_result

    def reshard(self) -> None:
        if self._training_state == TrainingState.FORWARD:
            if not self.reshard_after_forward:
                return
        logger.debug(
            "%s", self._with_fqn(f"YaFSDP::{self._training_state.name.lower()}_reshard")
        )
        self._to_sharded()
        self.data_buffer_ctx.release_event = (
            self.device_handle.current_stream().record_event()
        )
        self.data_buffer_ctx.owner = None

    def pre_forward(
        self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        logger.debug("%s", self._with_fqn("YaFSDP::pre_forward"))
        with record_function(self._with_fqn("YaFSDP::pre_forward")):
            self._training_state = TrainingState.FORWARD
            self.unshard(self.unshard_async_op)
            self.wait_for_unshard()
            args, kwargs = self._register_post_backward_hook(args, kwargs)
            return args, kwargs

    def post_forward(self, module: nn.Module, input: Any, output: Any) -> Any:
        logger.debug("%s", self._with_fqn("YaFSDP::post_forward"))
        with record_function(self._with_fqn("YaFSDP::post_forward")):
            if not is_bw():
                self.reshard()
                self._record_post_forward()
            self._training_state = TrainingState.IDLE
            return output

    def _record_post_forward(self) -> None:
        # Since a group has one pre-backward unshard for each forward call
        # before the backward, we record each usage (with multiplicity)
        post_forward_index = len(self.comm_ctx.post_forward_order)
        self.comm_ctx.post_forward_order.append(self)
        self._post_forward_indices.append(post_forward_index)

    def pre_backward(self, default_prefetch: bool, *unused: Any) -> None:
        if self._training_state == TrainingState.PRE_BACKWARD:
            return
        logger.debug("%s", self._with_fqn("YaFSDP::pre_backward"))
        with record_function(self._with_fqn("YaFSDP::pre_backward")):
            self._training_state = TrainingState.PRE_BACKWARD
            self.unshard(self.unshard_async_op)
            self.wait_for_unshard()
            if default_prefetch:
                self._backward_prefetch()

    def post_backward(self, *unused: Any) -> None:  # noqa: PLR0912
        # This method should be idempotent and safe to call even when this
        # FSDP parameter group was not used in backward (should be a no-op)
        logger.debug("%s", self._with_fqn("YaFSDP::post_backward"))
        self._training_state = TrainingState.POST_BACKWARD
        if (grad_buffer_ctx := self.grad_buffer_ctx) is not None:
            if (owner := grad_buffer_ctx.owner) is not None and owner is not self:
                raise RuntimeError(
                    f"{self} tried to acquire its gradient buffer, but it is in use by {owner}."
                )
            if (release_event := grad_buffer_ctx.release_event) is not None:
                self.device_handle.current_stream().wait_event(release_event)
                grad_buffer_ctx.release_event = None
            grad_buffer_ctx.owner = self
        with record_function(self._with_fqn("FSDP::post_backward_accumulate")):
            fsdp_params_with_grad_and_no_accumulated_grad: list[YaFSDPParam] = []
            unsharded_grads_with_no_accumulated_grad: list[torch.Tensor] = []
            fsdp_params_with_grad_and_accumulated_grad: list[YaFSDPParam] = []
            unsharded_grads_with_accumulated_grad: list[torch.Tensor] = []
            for fsdp_param in self.fsdp_params:
                if (
                    fsdp_param.unsharded_param.grad is not None
                    and fsdp_param.unsharded_accumulated_grad is None
                ):
                    fsdp_params_with_grad_and_no_accumulated_grad.append(fsdp_param)
                    unsharded_grads_with_no_accumulated_grad.append(
                        fsdp_param.unsharded_grad_data
                    )
                    fsdp_param.unsharded_param.grad = None
                    fsdp_param.unsharded_accumulated_grad = (
                        fsdp_param._unsharded_accumulated_grad
                    )
                elif fsdp_param.unsharded_param.grad is not None:
                    fsdp_params_with_grad_and_accumulated_grad.append(fsdp_param)
                    unsharded_grads_with_accumulated_grad.append(
                        fsdp_param.unsharded_grad_data
                    )
                    fsdp_param.unsharded_param.grad = None
            if len(fsdp_params_with_grad_and_no_accumulated_grad) != 0:
                torch._foreach_copy_(
                    [
                        cast("torch.Tensor", fsdp_param._unsharded_accumulated_grad)
                        for fsdp_param in fsdp_params_with_grad_and_no_accumulated_grad
                    ],
                    unsharded_grads_with_no_accumulated_grad,
                )
            if len(fsdp_params_with_grad_and_accumulated_grad) != 0:
                torch._foreach_add_(
                    [
                        cast("torch.Tensor", fsdp_param._unsharded_accumulated_grad)
                        for fsdp_param in fsdp_params_with_grad_and_accumulated_grad
                    ],
                    unsharded_grads_with_accumulated_grad,
                )
        with record_function(self._with_fqn("YaFSDP::post_backward_reshard")):
            if not self.reduce_grads:
                if self.reshard_after_backward:
                    self.reshard()
                return
            fsdp_params_with_grad: list[YaFSDPParam] = []
            for fsdp_param in self.fsdp_params:
                if fsdp_param.unsharded_accumulated_grad is not None:
                    fsdp_params_with_grad.append(fsdp_param)
                    fsdp_param.unsharded_accumulated_grad = None
            if self.reshard_after_backward:
                self.reshard()
        if len(fsdp_params_with_grad) == 0:
            if grad_buffer_ctx is not None:
                grad_buffer_ctx.release_event = (
                    self.device_handle.current_stream().record_event()
                )
                grad_buffer_ctx.owner = None
            return
        assert grad_buffer_ctx is not None
        logger.debug("%s", self._with_fqn("YaFSDP::post_backward_reduce"))
        with record_function(self._with_fqn("YaFSDP::post_backward_reduce")):
            reduce_scatter_input: dict[torch.dtype | None, torch.Tensor] = {}
            for all_gather_dtype in self._all_gather_dtype_to_fsdp_params:
                reduce_scatter_input[all_gather_dtype] = cast(
                    "torch.Tensor", (self._reduce_scatter_input[all_gather_dtype])
                )
            for all_gather_dtype in self._all_gather_dtype_to_fsdp_params:
                self._post_reduce_event = reduce_scatter(
                    self,
                    fsdp_params_with_grad,
                    cast(
                        "torch.Tensor",
                        self._padded_sharded_param_grad[all_gather_dtype],
                    ),
                    reduce_scatter_input[all_gather_dtype],
                    self._reduce_scatter_process_group,
                    self.comm_ctx.reduce_scatter_stream,
                    self.orig_dtype,
                    self.param_dtype,
                    self.reduce_dtype,
                    self.device,
                    self.gradient_divide_factor,
                    self.force_sum_reduction_for_comms,
                    self.mp_policy.bit32_acc_for_bit16_reduce_scatter,
                    cast("YaFSDPBufferContext", self.grad_buffer_ctx).yccl_handle,
                )
            grad_buffer_release_event = self._post_reduce_event
            grad_buffer_ctx.release_event = grad_buffer_release_event
            grad_buffer_ctx.owner = None

    def finalize_backward(self) -> None:
        self._wait_for_post_backward()
        if self._all_gather_result is not None:
            # If there was a mistargeted unshard without a corresponding wait,
            # then we wait here and clear the unshard
            logger.debug("%s", self._with_fqn("YaFSDP::wait_for_mistargeted_unshard"))
            self._all_gather_result.wait()
            self._all_gather_result = None
            self.data_buffer_ctx.owner = None
        self._post_forward_indices.clear()
        self.is_all_gather_input_set = False

    def _wait_for_post_backward(self) -> None:
        if self._post_reduce_event is not None:
            self.device_handle.current_stream().wait_event(self._post_reduce_event)
            self._post_reduce_event = None
            assert self.grad_buffer_ctx is not None
            assert self.grad_buffer_ctx.owner is None
            self.grad_buffer_ctx.release_event = None

    def _backward_prefetch(self) -> None:
        if not self._post_forward_indices:
            # Can be cleared if running multiple `backward`s
            return
        curr_index = self._post_forward_indices.pop()
        if (target_index := curr_index - 1) < 0:
            return
        # Prefetch naively using the reverse post-forward order, which may
        # have mistargeted prefetches if not all modules used in forward
        # are used in this backward
        target_fsdp_param_group = self.comm_ctx.post_forward_order[target_index]
        if target_fsdp_param_group.data_buffer_ctx is not self.data_buffer_ctx:
            self._prefetch_unshard(target_fsdp_param_group, "backward")

    @staticmethod
    def _prefetch_unshard(
        target_fsdp_param_group: "YaFSDPParamGroup", pass_type: str
    ) -> None:
        if pass_type == "backward":
            training_state = TrainingState.PRE_BACKWARD
        elif pass_type == "forward":
            training_state = TrainingState.FORWARD
        else:
            raise ValueError(f"Unknown pass type: {pass_type}")
        target_fqn = target_fsdp_param_group._module_fqn
        logger.debug("%s", f"YaFSDP::{pass_type}_prefetch for {target_fqn}")
        with (
            record_function(f"YaFSDP::{pass_type}_prefetch for {target_fqn}"),
            target_fsdp_param_group.use_training_state(training_state),
        ):
            async_op = target_fsdp_param_group.unshard_async_op
            target_fsdp_param_group.unshard(async_op)

    # Utilities #
    def _to_sharded(self) -> None:
        if not self.is_sharded:
            for fsdp_param in self.fsdp_params:
                fsdp_param.to_sharded()
            self._sharded_state = ShardedState.SHARDED

    def _to_unsharded(self) -> None:
        if not self.is_unsharded:
            for fsdp_param in self.fsdp_params:
                fsdp_param.to_unsharded()
            self._sharded_state = ShardedState.UNSHARDED

    @property
    def is_sharded(self) -> bool:
        return self._sharded_state == ShardedState.SHARDED

    @property
    def is_unsharded(self) -> bool:
        return self._sharded_state == ShardedState.UNSHARDED

    @contextlib.contextmanager
    def use_training_state(
        self, training_state: TrainingState
    ) -> Generator[None, Any, None]:
        old_training_state = self._training_state
        self._training_state = training_state
        try:
            yield
        finally:
            self._training_state = old_training_state

    # Hook Registration #
    def _register_post_backward_hook(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if not torch.is_grad_enabled():
            return args, kwargs
        args_list, args_spec = tree_flatten(args)
        kwargs_list, kwargs_spec = tree_flatten(kwargs)
        args_kwargs_list = list(args_list) + list(kwargs_list)
        inp_tensor_indices: list[int] = []
        inp_tensors: list[torch.Tensor] = []
        for i, obj in enumerate(args_kwargs_list):
            if torch.is_tensor(obj) and obj.requires_grad:
                inp_tensor_indices.append(i)
                inp_tensors.append(obj)
        if len(inp_tensors) == 0:
            return (args, kwargs)  # no tensors that require gradients
        inp_tensors = RegisterPostBackwardFunction.apply(self, *inp_tensors)
        for inp_tensor_idx, inp_tensor in zip(
            inp_tensor_indices, inp_tensors, strict=True
        ):
            args_kwargs_list[inp_tensor_idx] = inp_tensor
        args_list = args_kwargs_list[: len(args_list)]
        kwargs_list = args_kwargs_list[len(args_list) :]
        args = tree_unflatten(args_list, args_spec)
        kwargs = tree_unflatten(kwargs_list, kwargs_spec)
        return args, kwargs

    def _register_state_dict_hooks(self) -> None:
        num_pre_save_hooks = len(self._module_to_pre_save_state_dict_hook_handle)
        num_pre_load_hooks = len(self._module_to_pre_load_state_dict_hook_handle)
        if num_pre_save_hooks != num_pre_load_hooks:
            raise AssertionError(
                f"Pre-save: {num_pre_save_hooks} pre-load: {num_pre_load_hooks}"
            )
        if num_pre_save_hooks > 0:
            return  # already registered
        modules_with_fsdp_params: set[nn.Module] = {
            fsdp_param._module_info.module for fsdp_param in self.fsdp_params
        }

        def to_sharded_hook(*args: Any, **kwargs: Any) -> None:
            self._to_sharded()

        for module in modules_with_fsdp_params:
            self._module_to_pre_save_state_dict_hook_handle[module] = (
                module.register_state_dict_pre_hook(to_sharded_hook)
            )
            self._module_to_pre_load_state_dict_hook_handle[module] = (
                module._register_load_state_dict_pre_hook(to_sharded_hook)
            )

    # Properties #
    @property
    def reshard_after_forward(self) -> bool:
        return self.post_forward_mesh_info is not None

    @property
    def _all_gather_process_group(self) -> dist.ProcessGroup:
        if not isinstance(self.mesh_info, FSDPMeshInfo):
            raise AssertionError(
                f"Expected mesh_info to be FSDPMeshInfo, got {type(self.mesh_info)}"
            )
        return self.mesh_info.shard_process_group

    @property
    def _reduce_scatter_process_group(self) -> dist.ProcessGroup:
        if not isinstance(self.mesh_info, FSDPMeshInfo):
            raise AssertionError(
                f"Expected mesh_info to be FSDPMeshInfo, got {type(self.mesh_info)}"
            )
        return self.mesh_info.shard_process_group

    def _with_fqn(self, label: str) -> str:
        if self._module_fqn:
            return f"{label} ({self._module_fqn})"
        return label

    def __repr__(self) -> str:
        return f"YaFSDPParamGroup(fqn={self._module_fqn})"

    def _validate_no_meta_params(self) -> None:
        param_names_on_meta = [
            fsdp_param._param_fqn
            for fsdp_param in self.fsdp_params
            if fsdp_param.sharded_param.device.type == "meta"
        ]
        if param_names_on_meta:
            raise RuntimeError(
                "FSDP parameters should be materialized from meta device before training, "
                f"but the following were still on meta device: {param_names_on_meta}\n"
                "For example, call module.to_empty(device) to materialize to device and "
                "call module.reset_parameters() on each module to initialize values."
            )


def _get_param_module_infos(
    params: list[nn.Parameter], modules: tuple[nn.Module, ...]
) -> list[ParamModuleInfo]:
    params_set = set(params)
    param_to_module_info: dict[nn.Parameter, ParamModuleInfo] = {}
    for module in modules:
        for _, submodule in module.named_modules(remove_duplicate=False):
            for param_name, param in _named_parameters_with_duplicates(
                submodule, recurse=False
            ):
                if param in params_set:
                    if param not in param_to_module_info:
                        param_to_module_info[param] = ParamModuleInfo(
                            submodule, param_name
                        )
                    else:
                        param_to_module_info[param].shared_modules.append(submodule)
                        param_to_module_info[param].shared_param_names.append(
                            param_name
                        )
    if len(param_to_module_info) != len(params):
        raise AssertionError(f"Some parameters are not in the module tree of {module}")
    return [param_to_module_info[param] for param in params]


class RegisterPostBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, param_group: YaFSDPParamGroup, *inputs: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        ctx.param_group = param_group
        return inputs

    @staticmethod
    def backward(
        ctx: Any, *grads: torch.Tensor
    ) -> tuple[None, *tuple[torch.Tensor, ...]]:
        ctx.param_group.post_backward()
        return (None, *grads)
