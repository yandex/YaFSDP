import logging
import operator
import warnings
from dataclasses import dataclass
from functools import reduce
from typing import Any, cast

import torch
import torch.distributed.tensor._random as random
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    MetadataIndex,
    TensorProperties,
)
from torch.distributed.checkpoint.planner import (
    TensorWriteData,
    WriteItem,
    WriteItemType,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._ops._math_ops import _NormPartial
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
from torch.distributed.tensor.placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
)
from torch.optim.optimizer import (
    _foreach_supported_types as _optim_foreach_supported_types,
)
from torch.utils._foreach_utils import (
    _foreach_supported_types as _util_foreach_supported_types,
)

logger = logging.getLogger("ya_fsdp")


@dataclass(frozen=True)
class RaggedShard:
    local_numel: int
    global_offset: int
    shard_numels: tuple[int, ...]

    def __repr__(self) -> str:
        return f"RaggedShard(local_numel={self.local_numel}, global_offset={self.global_offset})"

    def __str__(self) -> str:
        return f"RS({self.local_numel}, {self.global_offset})"


class RaggedShardDTensor(torch.Tensor):
    _local_tensor: torch.Tensor
    _spec: DTensorSpec
    __slots__ = ["_local_tensor", "_spec"]

    @staticmethod
    def __new__(
        cls,
        local_tensor: torch.Tensor,
        spec: DTensorSpec,
        *,
        requires_grad: bool,
    ) -> "RaggedShardDTensor":
        if local_tensor.requires_grad and not requires_grad:
            warnings.warn(
                "To construct DTensor from torch.Tensor, it's recommended to "
                "use local_tensor.detach() and make requires_grad consistent.",
                stacklevel=1,
            )

        # new method instruct wrapper tensor from local_tensor and add
        # placement spec, it does not do actual distribution
        assert spec.tensor_meta is not None, "TensorMeta should not be None!"
        r = torch.Tensor._make_wrapper_subclass(
            cls,
            spec.tensor_meta.shape,
            strides=spec.tensor_meta.stride,
            dtype=local_tensor.dtype,
            device=local_tensor.device,
            layout=local_tensor.layout,
            requires_grad=requires_grad,
        )

        r._spec = spec
        r._local_tensor = local_tensor
        return r

    def __repr__(self) -> str:  # type: ignore[override]
        return f"RaggedShardDTensor(local_tensor={self._local_tensor}, device_mesh={self._spec.mesh}, placements={self._spec.placements})"

    def __tensor_flatten__(self) -> Any:
        return ["_local_tensor"], (self._spec, self.requires_grad)

    @staticmethod
    def __tensor_unflatten__(
        inner_tensors: Any, flatten_spec: Any, outer_size: Any, outer_stride: Any
    ) -> Any:
        assert flatten_spec is not None, (
            "Expecting spec to be not None from `__tensor_flatten__` return value!"
        )
        local_tensor = inner_tensors["_local_tensor"]
        spec, requires_grad = flatten_spec
        unflatten_tensor_meta = TensorMeta(
            shape=outer_size,
            stride=outer_stride,
            dtype=spec.tensor_meta.dtype,
        )
        unflatten_spec = DTensorSpec(
            spec.mesh,
            spec.placements,
            tensor_meta=unflatten_tensor_meta,
        )
        return RaggedShardDTensor(
            local_tensor,
            unflatten_spec,
            requires_grad=requires_grad,
        )

    @classmethod
    def __torch_dispatch__(  # type: ignore[override]  # noqa: PLR0911, PLR0912, PLR0915
        cls, func: Any, types: Any, args: Any = (), kwargs: Any = None
    ) -> Any:
        if func == torch.ops.aten.detach.default:
            (arg,) = args
            assert len(kwargs) == 0
            return RaggedShardDTensor(
                arg._local_tensor,
                arg._spec,
                requires_grad=False,
            )
        if func == torch.ops.aten.clone.default:
            (arg,) = args
            assert len(kwargs) == 0
            return RaggedShardDTensor(
                arg._local_tensor.clone(),
                arg._spec,
                requires_grad=False,
            )
        elif func == torch.ops.aten.copy_.default:
            arg1, arg2 = args
            func(arg1._local_tensor, arg2._local_tensor, **kwargs)
            return arg1
        elif func in (
            torch.ops.aten.empty_like.default,
            torch.ops.aten.zeros_like.default,
            torch.ops.aten.new_zeros.default,
            torch.ops.aten.new_empty.default,
        ):
            arg, *args = args
            return RaggedShardDTensor(
                func(arg._local_tensor, *args, **kwargs),
                arg._spec,
                requires_grad=arg.requires_grad,
            )
        elif func in (
            torch.ops.aten.zero_.default,
            torch.ops.aten.mul_.Tensor,
            torch.ops.aten.fill_.Scalar,
        ):
            arg, *args = args
            func(arg._local_tensor, *args, **kwargs)
            return arg
        elif func in (torch.ops.aten.normal_.default, torch.ops.aten.uniform_.default):
            arg, *args = args
            assert random._rng_tracker is not None
            rng_context = random._rng_tracker._distribute_region(
                DTensorSpec(
                    arg._spec.mesh,
                    (Shard(0), arg._spec.placements[1:]),
                    arg._spec.tensor_meta,
                )
            )
            with rng_context:
                func(arg._local_tensor, *args, **kwargs)
            return arg
        elif func == torch.ops.aten._foreach_sqrt.default:
            (args,) = args
            local_args = func([arg._local_tensor for arg in args], **kwargs)
            assert all(not arg.requires_grad for arg in args)
            return [
                RaggedShardDTensor(local_arg, arg._spec, requires_grad=False)
                for arg, local_arg in zip(args, local_args, strict=True)
            ]
        elif func == torch.ops.aten._foreach_norm.Scalar:
            args, norm_type = args
            local_args = func([arg._local_tensor for arg in args], norm_type, **kwargs)
            assert all(not arg.requires_grad for arg in args)
            return [
                DTensor(
                    local_arg,
                    DTensorSpec(
                        arg._spec.mesh,
                        tuple(
                            (
                                _NormPartial(norm_type)
                                if isinstance(placement, RaggedShard | Shard)
                                else placement
                            )
                            for placement in arg._spec.placements
                        ),
                        TensorMeta(torch.Size(), (), arg._spec.tensor_meta.dtype),
                    ),
                    requires_grad=False,
                )
                for arg, local_arg in zip(args, local_args, strict=True)
            ]
        elif func == torch.ops.aten._foreach_zero_.default:
            (args,) = args
            func([arg._local_tensor for arg in args], **kwargs)
        elif func in (
            torch.ops.aten._foreach_add_.Scalar,
            torch.ops.aten._foreach_mul_.Scalar,
            torch.ops.aten._foreach_div_.Scalar,
        ):
            args, scale = args
            func([arg._local_tensor for arg in args], scale, **kwargs)
        elif func == torch.ops.aten._foreach_div_.ScalarList:
            args, scalar_list = args
            func([arg._local_tensor for arg in args], scalar_list, **kwargs)
        elif func == torch.ops.aten._foreach_mul_.Tensor:
            args, tensor = args
            if isinstance(tensor, DTensor):
                assert tensor._spec.placements == (Replicate(),)
                tensor = tensor._local_tensor
            func([arg._local_tensor for arg in args], tensor, **kwargs)
        elif func == torch.ops.aten._foreach_add_.List:
            args1, args2 = args
            func(
                [arg._local_tensor for arg in args1],
                [arg._local_tensor for arg in args2],
                **kwargs,
            )
        elif func == torch.ops.aten._foreach_lerp_.Scalar:
            args1, args2, scalar = args
            func(
                [arg._local_tensor for arg in args1],
                [arg._local_tensor for arg in args2],
                scalar,
                **kwargs,
            )
        elif func in (
            torch.ops.aten._foreach_addcmul_.Scalar,
            torch.ops.aten._foreach_addcdiv_.Scalar,
        ):
            args1, args2, args3, scalar = args
            func(
                [arg._local_tensor for arg in args1],
                [arg._local_tensor for arg in args2],
                [arg._local_tensor for arg in args3],
                scalar,
                **kwargs,
            )
        elif func == torch.ops.aten._foreach_addcdiv_.ScalarList:
            args1, args2, args3, scalar_list = args
            func(
                [arg._local_tensor for arg in args1],
                [arg._local_tensor for arg in args2],
                [arg._local_tensor for arg in args3],
                scalar_list,
                **kwargs,
            )
        elif func == torch.ops.aten._fused_adam_.default:
            args1, args2, args3, args4, args5, args6 = args
            assert len(args5) == 0
            func(
                [arg._local_tensor for arg in args1],
                [arg._local_tensor for arg in args2],
                [arg._local_tensor for arg in args3],
                [arg._local_tensor for arg in args4],
                args5,
                args6,
                **kwargs,
            )
        else:
            raise NotImplementedError(
                f"RaggedShardDTensor dispatch for {func} is not implemented yet."
            )

    def to_local(self) -> torch.Tensor:
        if torch.is_grad_enabled():
            raise NotImplementedError
        return self._local_tensor

    @property
    def device_mesh(self) -> DeviceMesh:
        return self._spec.mesh

    @property
    def placements(self) -> tuple[Placement, ...]:
        return self._spec.placements

    def _raise_if_contains_partial_placements(self) -> None:
        for placement in self._spec.placements:
            if not isinstance(placement, Partial):
                continue

            raise ValueError(
                "Any checkpointing related operations are not supported for "
                "DTensor with partial placements!"
            )

    def full_tensor(self) -> torch.Tensor:
        shard_numels = cast("RaggedShard", self._spec.placements[0]).shard_numels
        full_tensor = self._local_tensor.new_empty(size=(sum(shard_numels),))
        torch.distributed.all_gather(
            list(full_tensor.split(shard_numels)),
            self._local_tensor,
            group=self._spec.mesh.get_group(0),
        )
        if len(self._spec.placements) > 1:
            local_shape, _ = compute_local_shape_and_global_offset(
                self.shape, self._spec.mesh, (Replicate(), *self._spec.placements[1:])
            )
            return DTensor(
                full_tensor.view(local_shape),
                DTensorSpec(
                    self._spec.mesh,
                    (Replicate(), *self._spec.placements[1:]),
                    self._spec.tensor_meta,
                ),
                requires_grad=self.requires_grad,
            ).full_tensor()
        else:
            return full_tensor.view(self.size())

    def __create_write_items__(self, fqn: str, object: Any) -> list[WriteItem]:
        assert object is self
        self._raise_if_contains_partial_placements()
        ragged_shard_placement = cast("RaggedShard", self._spec.placements[0])
        if ragged_shard_placement.local_numel == 0:
            offsets = torch.Size((self.numel(),))
        else:
            _, global_offsets = compute_local_shape_and_global_offset(
                self.shape, self._spec.mesh, (Replicate(), *self._spec.placements[1:])
            )
            global_offset = sum(
                offset * stride
                for offset, stride in zip(global_offsets, self.stride(), strict=True)
            )
            offsets = torch.Size(
                (global_offset + ragged_shard_placement.global_offset,)
            )
        sizes = torch.Size((ragged_shard_placement.local_numel,))
        return [
            WriteItem(
                index=MetadataIndex(fqn, offsets),
                type=WriteItemType.SHARD,
                tensor_data=TensorWriteData(
                    chunk=ChunkStorageMetadata(offsets=offsets, sizes=sizes),
                    properties=TensorProperties.create_from_tensor(self._local_tensor),
                    size=torch.Size((reduce(operator.mul, self.size(), 1),)),
                ),
            )
        ]

    def __create_chunk_list__(self) -> list[ChunkStorageMetadata]:
        self._raise_if_contains_partial_placements()
        if hasattr(self._local_tensor, "__create_chunk_list__"):
            return cast(
                "list[ChunkStorageMetadata]", self._local_tensor.__create_chunk_list__()
            )
        elif isinstance(self._local_tensor, torch.Tensor):
            ragged_shard_placement = cast("RaggedShard", self._spec.placements[0])
            if ragged_shard_placement.local_numel == 0:
                offsets = torch.Size((self.numel(),))
            else:
                _, global_offsets = compute_local_shape_and_global_offset(
                    self.shape,
                    self._spec.mesh,
                    (Replicate(), *self._spec.placements[1:]),
                )
                global_offset = sum(
                    offset * stride
                    for offset, stride in zip(
                        global_offsets, self.stride(), strict=True
                    )
                )
                offsets = torch.Size(
                    (global_offset + ragged_shard_placement.global_offset,)
                )
            sizes = torch.Size((ragged_shard_placement.local_numel,))
            return [ChunkStorageMetadata(offsets=offsets, sizes=sizes)]
        else:
            raise RuntimeError("Unsupported tensor type!")

    def __get_tensor_shard__(self, index: int) -> torch.Tensor:
        self._raise_if_contains_partial_placements()
        if hasattr(self._local_tensor, "__get_tensor_shard__"):
            return cast("torch.Tensor", self._local_tensor.__get_tensor_shard__(index))
        elif isinstance(self._local_tensor, torch.Tensor):
            return self._local_tensor
        else:
            raise RuntimeError("Unsupported tensor type!")


torch.serialization.add_safe_globals([RaggedShardDTensor, RaggedShard])

# Append RaggedShardDTensor to the list of supported types for foreach implementation for optimizer
# and clip_grad_norm_ so that we will try to use foreach over the for-loop implementation on CUDA.
if RaggedShardDTensor not in _optim_foreach_supported_types:
    _optim_foreach_supported_types.append(RaggedShardDTensor)

if RaggedShardDTensor not in _util_foreach_supported_types:  # type: ignore[comparison-overlap]
    _util_foreach_supported_types.append(RaggedShardDTensor)  # type: ignore[arg-type]
