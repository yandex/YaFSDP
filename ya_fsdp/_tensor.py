import logging
from dataclasses import dataclass
from typing import cast

import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import _is_inplace_op, _is_out_variant_op
from torch.distributed.tensor.placement_types import Placement, Shard, _StridedShard
from torch.optim.optimizer import _foreach_supported_types as _optim_foreach_supported_types
from torch.utils._foreach_utils import _foreach_supported_types as _util_foreach_supported_types

try:
    from torch.utils import _cxx_pytree as pytree
except ImportError:
    from torch.utils import _pytree as pytree

aten = torch.ops.aten
logger = logging.getLogger("ya_fsdp")


@dataclass(frozen=True)
class YaFSDPShard(Placement):
    def __repr__(self) -> str:
        return "YaFSDPShard()"

    def __str__(self) -> str:
        return "YaFSDPS()"


@dataclass(frozen=True, kw_only=True)
class _YaFSDPStridedShard(YaFSDPShard):
    split_factor: int

    def __repr__(self) -> str:
        return f"_YaFSDPStridedShard(sf={self.split_factor})"

    def __str__(self) -> str:
        return f"_YaFSDPS({self.split_factor})"


@dataclass(kw_only=True)
class YaFSDPDTensorSpec(DTensorSpec):
    local_numel: int
    global_offset: int

    __hash__ = DTensorSpec.__hash__


class YaFSDPDTensor(DTensor):
    _spec: YaFSDPDTensorSpec

    @staticmethod
    @torch._disable_dynamo
    def __new__(
        cls,
        local_tensor: torch.Tensor,
        spec: YaFSDPDTensorSpec,
        *,
        requires_grad: bool,
    ) -> "DTensor":
        return DTensor.__new__(cls, local_tensor, spec, requires_grad=requires_grad)

    def __repr__(self):
        return f"YaFSDPDTensor(local_tensor={self._local_tensor}, device_mesh={self._spec.mesh}, placements={self._spec.placements}, local_numel={self._spec.local_numel}, global_offset={self._spec.global_offset})"

    @staticmethod
    def __tensor_unflatten__(inner_tensors, flatten_spec, outer_size, outer_stride):
        assert flatten_spec is not None, "Expecting spec to be not None from `__tensor_flatten__` return value!"
        local_tensor = inner_tensors["_local_tensor"]
        spec, requires_grad = flatten_spec
        unflatten_tensor_meta = TensorMeta(
            shape=outer_size,
            stride=outer_stride,
            dtype=spec.tensor_meta.dtype,
        )
        unflatten_spec = YaFSDPDTensorSpec(
            spec.mesh,
            spec.placements,
            tensor_meta=unflatten_tensor_meta,
            local_numel=spec.local_numel,
            global_offset=spec.global_offset,
        )
        return YaFSDPDTensor(
            local_tensor,
            unflatten_spec,
            requires_grad=requires_grad,
        )

    @classmethod
    @torch._disable_dynamo
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        op_call = func
        kwargs = kwargs or {}
        flat_args_kwargs = (*pytree.tree_flatten(args)[0], *pytree.tree_flatten(kwargs)[0])
        arg_to_spec = {arg: arg._spec for arg in flat_args_kwargs if isinstance(arg, YaFSDPDTensor)}
        for arg, spec in arg_to_spec.items():
            arg._spec = DTensorSpec(
                spec.mesh,
                (
                    (
                        Shard(0)
                        if isinstance((first_placement := (placements := spec.placements)[0]), YaFSDPShard)
                        else (
                            _StridedShard(0, first_placement.split_factor)
                            if isinstance(first_placement, _YaFSDPStridedShard)
                            else first_placement
                        )
                    ),
                    *placements[1:],
                ),
                tensor_meta=spec.tensor_meta,
            )

        # operators that does not need to go through sharding propagation
        if op_call in DTensor._op_dispatcher._custom_op_handlers:
            raise NotImplementedError

        # extract local tensor and sharding infos to a OpInfo
        op_info = DTensor._op_dispatcher.unwrap_to_op_info(op_call, args, kwargs)
        logger.debug("Dispatching op_call: %s", op_info.schema)

        DTensor._op_dispatcher.sharding_propagator.propagate(op_info)
        output_sharding = op_info.output_sharding
        output_sharding.output_spec = None if output_sharding.output_spec is None else next(iter(arg_to_spec.values()))
        for arg, spec in arg_to_spec.items():
            arg._spec = spec
        logger.debug("output_sharding for %s: %s", op_call, output_sharding)
        assert output_sharding is not None, "output sharding should not be None"

        mesh = op_info.compute_mesh
        if mesh.get_coordinate() is not None:
            # computation that happens in the current rank of the mesh, normal case
            if output_sharding.needs_redistribute:
                raise NotImplementedError

            local_tensor_args = (
                pytree.tree_unflatten(cast(list[object], op_info.local_args), op_info.args_tree_spec)
                if op_info.args_tree_spec
                else op_info.local_args
            )

            # run local op computation with potentially modified args/kwargs
            local_tensor_args = cast(tuple[object, ...], local_tensor_args)
            if op_call in DTensor._op_dispatcher._random_ops:
                raise NotImplementedError
            else:
                # normal case, run local sharded op computation
                local_results = op_call(*local_tensor_args, **op_info.local_kwargs)

        else:
            raise NotImplementedError

        if output_sharding.output_spec is None:
            if op_call == aten.equal.default:
                raise NotImplementedError

        if _is_inplace_op(op_call):
            # inplace op should return self instead of re-wrapping
            if output_sharding.output_spec is not None:
                return args[0]
            else:
                return None
        elif _is_out_variant_op(op_call):
            raise NotImplementedError
        else:
            res = local_results
            spec = output_sharding.output_spec
            if isinstance(res, torch.Tensor):
                if spec is not None:
                    assert isinstance(
                        spec, YaFSDPDTensorSpec
                    ), f"output spec does not match with output! Expected DTensorSpec, got {spec}."
                    return YaFSDPDTensor(res, spec, requires_grad=res.requires_grad)
                else:
                    # if output does not have a DTensorSpec due to specific ops, it must be a scalar tensor
                    assert res.ndim == 0, "output tensor should be scalar!"
                    return res
            elif isinstance(res, (list, tuple)):
                raise NotImplementedError
            else:
                # if the res contains only non tensor values (i.e. int/float/none), we simply return it
                # without rewrapping to DTensor.
                return res

    @property
    def local_numel(self) -> int:
        return self._spec.local_numel

    @property
    def global_offset(self) -> int:
        return self._spec.global_offset


torch.serialization.add_safe_globals(
    [
        YaFSDPDTensorSpec,
        YaFSDPDTensor,
        YaFSDPShard,
        _YaFSDPStridedShard,
    ]
)

if DTensor not in _optim_foreach_supported_types:
    _optim_foreach_supported_types.append(YaFSDPDTensor)

if DTensor not in _util_foreach_supported_types:
    _util_foreach_supported_types.append(YaFSDPDTensor)
