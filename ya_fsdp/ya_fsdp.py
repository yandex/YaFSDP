import gc
import time
from collections import OrderedDict
from enum import Enum
from typing import Any, Callable, Type

import torch
import torch.distributed as dist
import torch.distributed.fsdp.wrap
import torch.nn as nn
from torch.distributed.fsdp._init_utils import (
    _init_intra_node_process_group,
    _materialize_with_param_init_fn,
    _move_module_to_device,
    _need_to_materialize_module,
    _sync_module_states,
)

from .meta_param import MetaParam, convert_some_params_to_metaparams, materialize_params


class ReusableBufferViewState(Enum):
    NOT_READY = 1
    READY = 2
    IN_PROCESS = 3


class YaFSDP(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        zero_stage: int,
        param_dtype: torch.dtype,
        modules_to_wrap_with_names: list[tuple[nn.Module, str]],
        layer_norm_module_cls: Type[nn.Module],
        gradient_accumulation_steps: int,
        data_parallel_process_group: dist.ProcessGroup | None = None,
        intra_node_data_parallel_process_group: dist.ProcessGroup | None = None,
        model_parallel_process_group: dist.ProcessGroup | None = None,
        all_reduce_grads_across_model_parallel_group: bool = False,
        bit16_reduce_scatter: bool = False,
        bit32_acc_for_bit16_reduce_scatter: bool = False,
        hpz_first_layers_num: int = 0,
        output_layer_module_with_name: tuple[nn.Module, str] | None = None,
        sync_module_states: bool = False,
        param_init_fn: Callable | None = None,
        device_id: int | None = None,
    ):
        super().__init__()
        self._module = module
        self._zero_stage = zero_stage
        self._gradient_accumulation_steps = gradient_accumulation_steps
        self._device = torch.device(torch.cuda.current_device() if device_id is None else device_id)

        self._data_parallel_process_group = (
            dist.distributed_c10d._get_default_group()
            if data_parallel_process_group is None
            else data_parallel_process_group
        )
        self._intra_node_data_parallel_process_group = (
            _init_intra_node_process_group(torch.cuda.device_count())
            if intra_node_data_parallel_process_group is None
            else intra_node_data_parallel_process_group
        )
        self._model_parallel_process_group = model_parallel_process_group

        self._compute_stream = torch.cuda.current_stream()
        self._comm_stream = torch.cuda.Stream(priority=-1)
        self._hpz_all_gather_stream = torch.cuda.Stream(priority=-1) if hpz_first_layers_num > 0 else None

        target_module_to_kwargs = {
            m: {
                "comm_stream": self._comm_stream,
                "data_parallel_process_group": self._data_parallel_process_group,
                "intra_node_data_parallel_process_group": self._intra_node_data_parallel_process_group,
                "model_parallel_process_group": self._model_parallel_process_group,
                "is_hpz_layer": hpz_first_layers_num > idx,
                "hpz_all_gather_stream": self._hpz_all_gather_stream,
                "name": name,
                "sync_module_states": sync_module_states,
                "param_init_fn": param_init_fn,
                "layer_norm_module_cls": layer_norm_module_cls,
                "device": self._device,
            }
            for idx, (m, name) in enumerate(modules_to_wrap_with_names)
        }

        torch.distributed.fsdp.wrap._post_order_apply(
            self._module,
            fn=torch.distributed.fsdp.wrap._construct_wrap_fn(
                self._module,
                target_module_to_kwargs=target_module_to_kwargs,
                fsdp_fn=SuperTensorModule,
            ),
        )

        self._backwards_count = 0

        max_buffer_size = 0
        sum_shards = 0
        max_shard = 0
        # Meta stored in every process
        # conversions. Stored in state dict as well.
        self._meta_info = {
            "main_parameter_shard_bounaries": [],  # keeps fragmentation (by modules) of main_params (By shard sizes)
            "main_parameter_padded_numels": [],  # keeps padded numels of each module
            "main_shards_fqns": [],  # keep List[List]. Each list keeps ordered fqns of this module
            "main_shards_numels": [],  # keep List[List]. Each list keeps ordered numels of params of this module
            "layernorm_fqns": [],  # keep List[List]. Each list keeps ordered fqns of lns of this module
            "layernorm_numels": [],  # keep List[List]. Each list keeps ordered numels of ln of this composed module
            "main_parameter_orig_shapes": [],  # keep List[List]. Each list keeps ordered orig shapes of params
            "layernorm_orig_shapes": [],  # keep List[List]. Each list keeps ordered orig shapes of ln params
        }

        self._super_tensors: list[SuperTensor] = []
        unflattened_params = []

        self._meta_info["layernorm_fqns"] = []
        self._meta_info["layernorm_numels"] = []
        self._meta_info["layernorm_orig_shapes"] = []

        for module_ in self._module.modules():
            if output_layer_module_with_name is not None and module_ is output_layer_module_with_name[0]:
                is_meta_module, _ = _need_to_materialize_module(module_, set(), set())
                if sync_module_states and is_meta_module:
                    _materialize_with_param_init_fn(
                        root_module=module_,
                        param_init_fn=param_init_fn,
                        ignored_modules=set(),
                    )
                _move_module_to_device(
                    module=module_,
                    ignored_params=set(),
                    ignored_buffers=set(),
                    device_from_device_id=self._device,
                )
                if sync_module_states:
                    _sync_module_states(
                        params=list(module_.parameters()),
                        buffers=list(module_.buffers()),
                        process_group=self._intra_node_data_parallel_process_group,
                    )

                layer_unflattened_params = convert_some_params_to_metaparams(
                    module_, output_layer_module_with_name[1], lambda _: True
                )
                unflattened_params.extend(layer_unflattened_params)
                self._meta_info["layernorm_fqns"].append(
                    [f"{param.parent_module_name}.{param.name}" for param in layer_unflattened_params]
                )
                self._meta_info["layernorm_numels"].append([param.numel for param in layer_unflattened_params])
                self._meta_info["layernorm_orig_shapes"].append([param.shape for param in layer_unflattened_params])
            elif isinstance(module_, SuperTensorModule):
                self._super_tensors.append(module_.super_tensor)

                max_buffer_size = max(max_buffer_size, module_.super_tensor.padded_numel)
                sum_shards += module_.super_tensor.shard_size
                max_shard = max(max_shard, module_.super_tensor.shard_size)
                # Add meta
                self._meta_info["main_parameter_shard_bounaries"].append(module_.super_tensor.shard_size)
                self._meta_info["main_parameter_padded_numels"].append(module_.super_tensor.padded_numel)

                self._meta_info["main_shards_fqns"].append(
                    [f"{module_.name}.{fqname}" for fqname in module_.module_internal_fqns]
                )
                self._meta_info["main_shards_numels"].append(module_.module_internal_numels)
                self._meta_info["main_parameter_orig_shapes"].append(
                    [list(x) for x in module_.super_tensor.params_shapes]
                )

                layer_unflattened_params = convert_some_params_to_metaparams(
                    module_.root_module, module_.name, lambda _: True
                )
                unflattened_params.extend(layer_unflattened_params)
                self._meta_info["layernorm_fqns"].append(
                    [f"{param.parent_module_name}.{param.name}" for param in layer_unflattened_params]
                )
                self._meta_info["layernorm_numels"].append([param.numel for param in layer_unflattened_params])
                self._meta_info["layernorm_orig_shapes"].append([param.shape for param in layer_unflattened_params])

        self._layernorm_supertensor = SuperTensor(
            unflattened_params,
            name="layernorm_supertensor",
            comm_stream=self._comm_stream,
            data_parallel_process_group=self._data_parallel_process_group,
            intra_node_data_parallel_process_group=self._intra_node_data_parallel_process_group,
            model_parallel_process_group=self._model_parallel_process_group,
            all_reduce_grads_across_model_parallel_group=all_reduce_grads_across_model_parallel_group,
        )
        self._super_tensors.append(self._layernorm_supertensor)

        self._all_buffers = []

        if not bit16_reduce_scatter:
            self._fp32_grad_buffer = ReusableBuffer(
                max(self._layernorm_supertensor.total_numel, max_buffer_size),
                device="cuda",
                dtype=torch.float,
                prepare_stream=self._comm_stream,
                process_stream=self._comm_stream,
            )
            self._all_buffers.append(self._fp32_grad_buffer)
        else:
            self._fp32_grad_buffer = None

        self._layernorm_weight_buffer = ReusableBuffer(
            self._layernorm_supertensor.padded_numel,
            device="cuda",
            dtype=param_dtype,
            prepare_stream=self._comm_stream,
            process_stream=self._compute_stream,
        )
        self._layernorm_grad_buffer = ReusableBuffer(
            self._layernorm_supertensor.padded_numel,
            device="cuda",
            dtype=param_dtype,
            prepare_stream=self._compute_stream,
            process_stream=self._comm_stream,
            fill_zeros=True,
        )
        self._all_buffers.extend([self._layernorm_weight_buffer, self._layernorm_grad_buffer])

        self._layernorm_supertensor.set_buffers(self._layernorm_weight_buffer, self._layernorm_grad_buffer)
        self._layernorm_supertensor.set_fp32_grad_buffer(self._fp32_grad_buffer)
        self._layernorm_parameter = torch.nn.Parameter(
            torch.empty(self._layernorm_supertensor.shard_size).float().cuda()
        )
        self._layernorm_parameter.grad = torch.zeros(self._layernorm_supertensor.shard_size).float().cuda()
        self._layernorm_parameter_bit16 = torch.empty(self._layernorm_supertensor.shard_size).to(param_dtype).cuda()
        self._layernorm_supertensor.set_weight_shard_buffers(
            self._layernorm_parameter_bit16, self._layernorm_parameter, 0
        )

        self._main_parameter = torch.nn.Parameter(torch.empty(sum_shards).float().cuda())
        self._main_parameter.grad = torch.zeros(sum_shards).float().cuda()

        self._main_parameter_bit16 = torch.empty(sum_shards).to(param_dtype).cuda()
        self._weight_buffers = [
            ReusableBuffer(
                max_buffer_size if zero_stage == 3 else self._super_tensors[i].padded_numel,
                device="cuda",
                dtype=param_dtype,
                prepare_stream=self._comm_stream,
                process_stream=self._compute_stream,
            )
            for i in range(2 if zero_stage == 3 else len(self._super_tensors))
        ]
        self._grad_buffers = [
            ReusableBuffer(
                max_buffer_size if zero_stage > 1 else self._super_tensors[i].padded_numel,
                device="cuda",
                dtype=param_dtype,
                prepare_stream=self._compute_stream,
                process_stream=self._comm_stream,
                fill_zeros=True,
            )
            for i in range(2 if zero_stage > 1 else len(self._super_tensors))
        ]
        self._all_buffers.extend(self._weight_buffers)
        self._all_buffers.extend(self._grad_buffers)

        shard_buffer_offset = 0
        for idx, module_ in enumerate(m for m in self._module.modules() if isinstance(m, SuperTensorModule)):
            module_.super_tensor.set_buffers(
                self._weight_buffers[idx % len(self._weight_buffers)],
                self._grad_buffers[idx % len(self._grad_buffers)],
            )
            module_.super_tensor.set_fp32_grad_buffer(self._fp32_grad_buffer)
            module_.super_tensor.set_weight_shard_buffers(
                self._main_parameter_bit16, self._main_parameter, shard_buffer_offset
            )
            if bit32_acc_for_bit16_reduce_scatter:
                module_.super_tensor.enable_bit32_acc_for_bit16_reduce_scatter()
            shard_buffer_offset += module_.super_tensor.shard_size

        self.register_forward_pre_hook(self._get_forward_pre_hook())
        self.register_forward_hook(self._layernorm_supertensor.get_forward_hook())

        self.register_full_backward_pre_hook(self._layernorm_supertensor.get_backward_pre_hook())

        gc.collect()
        torch.cuda.empty_cache()

    def _get_backward_hook(self):
        def backward_hook():
            self._backwards_count = (self._backwards_count + 1) % self._gradient_accumulation_steps
            if self._backwards_count > 0:
                if self._zero_stage > 1:
                    self._layernorm_grad_buffer.free()
                return

            for buffer in sorted(self._all_buffers, key=lambda buffer: buffer.last_time_processed):
                buffer.free()

            for super_tensor in self._super_tensors:
                if super_tensor.is_hpz_layer:
                    super_tensor.hpz_inited = False

            comm_wait_event = torch.cuda.Event()
            comm_wait_event.record(self._comm_stream)
            comm_wait_event.wait()

            divider = self._data_parallel_process_group.size() * self._gradient_accumulation_steps
            self._main_parameter.grad.data.div_(divider)
            self._layernorm_parameter.grad.data.div_(divider)

        return backward_hook

    def _init_grad_flows_and_set_hook(self):
        flows = []
        for super_tensor in self._super_tensors:
            super_tensor.init_weight_grad_flow()
            flows.append(super_tensor.weight_grad_flow)
        flows = GateGradFlow.apply(self._get_backward_hook(), *flows)

        for super_tensor, flow in zip(self._super_tensors, flows):
            super_tensor.weight_grad_flow = flow

    def _get_forward_pre_hook(self):
        def forward_pre_hook(module, input):
            if self._backwards_count == 0 and not (self._zero_stage == 2 and not torch.is_grad_enabled()):
                # In zero - 2 stage in eval mode we do not need to free buffers as they will be used further
                self._layernorm_parameter_bit16.data.copy_(self._layernorm_parameter)
                self._main_parameter_bit16.data.copy_(self._main_parameter)
                for buffer in sorted(self._all_buffers, key=lambda buffer: buffer.last_time_processed):
                    buffer.free()
                for super_tensor in self._super_tensors:
                    super_tensor.first_reduce_scatter = True

            self._init_grad_flows_and_set_hook()  # set up backward hook
            self._layernorm_supertensor.get_forward_pre_hook()(module, input)
            self._layernorm_supertensor.pre_forward(input)

        return forward_pre_hook

    def __getattr__(self, name: str) -> Any:
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self._module, name)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run the forward pass for the wrapped module."""
        return self._module(*args, **kwargs)

    def clip_grad_norm_(self, max_norm: float | int, norm_type: float | int = 2.0) -> torch.Tensor:
        assert norm_type in {2.0, 2}
        grad_norm = torch.linalg.vector_norm(
            torch.stack(
                [
                    torch.linalg.vector_norm(grad.detach(), norm_type, dtype=torch.float32)
                    for grad in [self._layernorm_parameter.grad, self._main_parameter.grad]
                ],
            ),
            norm_type,
            dtype=torch.float32,
        )
        total_norm = grad_norm**norm_type
        dist.all_reduce(total_norm, group=self._data_parallel_process_group)
        if self._model_parallel_process_group is not None and self._model_parallel_process_group.size() > 1:
            dist.all_reduce(total_norm, group=self._model_parallel_process_group)
        total_norm = total_norm ** (1 / norm_type)
        total_norm_coef = max_norm / (total_norm + 1e-6)
        total_norm_coef = torch.clamp(total_norm_coef, max=1.0)
        self._layernorm_parameter.grad.mul_(total_norm_coef)
        self._main_parameter.grad.mul_(total_norm_coef)
        return total_norm

    def local_state_dict(self):
        state_dict = super().state_dict()
        # Save meta on chief processes
        if self._data_parallel_process_group.rank() == 0:
            state_dict["meta"] = self._meta_info

        return state_dict

    def state_dict(self):
        state_dict = OrderedDict()
        for super_tensor in self._super_tensors:
            super_tensor.collect_module()
            if self._data_parallel_process_group.rank() != 0:
                continue
            for param in super_tensor.params:
                data = getattr(param.parent_module, param.name)
                state_dict[f"{param.parent_module_name}.{param.name}"] = data.detach().cpu()
        return state_dict

    def load_state_dict(self, state_dict):
        state_dict.pop("meta", None)
        super().load_state_dict(state_dict)

    def get_files_to_load(self):
        files_to_load = [
            "latest",
            "iteration_info",
        ]  # common meta
        files_to_load.append(f"reader_{torch.distributed.get_rank()}.pkl")  # reader
        model_parallel_rank = (
            0 if self._model_parallel_process_group is None else self._model_parallel_process_group.rank()
        )
        files_to_load.append(
            f"yafsdp_optim_{self._data_parallel_process_group.rank():05d}_{model_parallel_rank:02d}.pt"
        )
        files_to_load.append("lr_scheduler.pt")  # lr scheduler for all the same
        files_to_load.append(
            f"yafsdp_model_{self._data_parallel_process_group.rank():05d}_{model_parallel_rank:02d}.pt"
        )  # local "shard"
        return files_to_load

    def zero_grad(self, *args, **kwargs) -> None:
        set_to_none = False
        if args:
            assert len(args) == 1 and not kwargs
            set_to_none = args[0]
        elif kwargs:
            assert len(kwargs) == 1 and (key := "set_to_none") in kwargs and not args
            set_to_none = kwargs[key]
        if set_to_none:
            raise ValueError("YaFSDP doesn't support zero_grad(set_to_none=True).")
        super().zero_grad(set_to_none)


class ReusableBuffer:
    def __init__(self, size, dtype, device, prepare_stream, process_stream, fill_zeros=False):
        self._buffer = (
            torch.zeros(size, dtype=dtype, device=device)
            if fill_zeros
            else torch.empty(size, dtype=dtype, device=device)
        )
        self._owner = None

        self._prepared_event = None
        self._released_event = None

        self._active_child = None
        self._prev_child = None

        self.last_time_processed = 0

        self._main_prepare_stream = prepare_stream
        self._prepare_stream = prepare_stream
        self._process_stream = process_stream

    def set_prepare_stream(self, stream):
        self._prepare_stream = stream

    def unset_prepare_stream(self):
        self._prepare_stream = self._main_prepare_stream

    def get_prefix(self, real_size, padded_size, name=""):
        return ReusableBufferView(self, real_size, padded_size, name)

    def _set_owner(self, buffer_view, super_tensor):
        assert self._owner is None
        assert self._active_child is None
        self._owner = super_tensor
        self._active_child = buffer_view

    def is_gained(self, super_tensor):
        return self._owner == super_tensor

    # only in start_prepare we can change owner
    def start_prepare(self, buffer_view, super_tensor):
        if self.is_gained(super_tensor):
            return
        self._free()
        self._prepared_event = None
        self._released_event = None
        self._set_owner(buffer_view, super_tensor)

    def end_prepare(self, super_tensor):
        assert self._owner == super_tensor, f"{self._owner} != {super_tensor}"
        self._prepared_event = torch.cuda.Event()
        self._prepared_event.record(self._prepare_stream)

    def start_processing(self, super_tensor):
        assert self._owner == super_tensor, f"{self._owner} != {super_tensor}"
        assert self._prepared_event is not None
        self._prepared_event.wait(self._process_stream)
        self._released_event = None

    def end_processing(self, super_tensor):
        assert self._owner == super_tensor, f"{self._owner} != {super_tensor}"
        self._released_event = torch.cuda.Event()
        self._released_event.record(self._process_stream)
        self.last_time_processed = time.time()

    def _free(self):
        # we use self._prepare_stream because we want to free only after end_processing
        with torch.cuda.stream(self._prepare_stream):
            if self._owner:
                for hook in self._active_child._free_hooks:
                    # the hook implies a delayed call of something that will work in the process_stream
                    hook(self._process_stream)
                assert self._released_event is not None
                self._released_event.wait()
                self._owner.released_event = None
                self._owner.gathered = False
                self._owner = None
                self._prev_child = self._active_child
                self._active_child = None
        self._state = ReusableBufferViewState.NOT_READY

    def free(self):
        if self._active_child is not None:
            self._active_child.free()


class ReusableBufferView:
    def __init__(self, reusable_buffer, real_size, padded_size, name=""):
        self._name = name
        self._real_size = real_size
        self._size = padded_size
        self.reusable_buffer = reusable_buffer
        self._buffer_view = reusable_buffer._buffer.narrow(0, 0, padded_size)
        self._free_hooks = []
        self._state = ReusableBufferViewState.NOT_READY

    def get(self):
        return self._buffer_view

    def set_prepare_stream(self, stream):
        self.reusable_buffer.set_prepare_stream(stream)

    def unset_prepare_stream(self):
        self.reusable_buffer.unset_prepare_stream()

    # method to understand if we have zero grads
    def is_super_tensor_superset_of_previous_owner(self):
        if self.reusable_buffer._prev_child is None:
            return False
        return self._real_size >= self.reusable_buffer._prev_child._real_size

    def start_prepare(self, super_tensor):
        self.reusable_buffer.start_prepare(self, super_tensor)
        self._state = ReusableBufferViewState.NOT_READY

    def end_prepare(self, super_tensor):
        self.reusable_buffer.end_prepare(super_tensor)
        self._state = ReusableBufferViewState.READY

    def start_processing(self, super_tensor):
        assert self._state == ReusableBufferViewState.READY, f"real state is {self._state}"
        self.reusable_buffer.start_processing(super_tensor)
        self._state = ReusableBufferViewState.IN_PROCESS

    def end_processing(self, super_tensor):
        assert self._state == ReusableBufferViewState.IN_PROCESS, f"real state is {self._state}"
        self.last_time_gained = time.time()
        self.reusable_buffer.end_processing(super_tensor)
        self._state = ReusableBufferViewState.READY

    def is_gained(self, super_tensor):
        return self.reusable_buffer.is_gained(super_tensor)

    def free(self):
        self.reusable_buffer._free()
        self._state = ReusableBufferViewState.NOT_READY

    def register_free_hook(self, hook):
        self._free_hooks.append(hook)


# this function guaranties that gate gradient will be computed before input gradient
class GateGradFlow(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hook, *inputs):
        ctx.set_materialize_grads(False)
        ctx.hook = hook
        return inputs

    @staticmethod
    def backward(ctx, *grads):
        ctx.hook()
        return None, *grads


class SuperTensor:
    def __init__(
        self,
        params,
        comm_stream,
        name,
        data_parallel_process_group: torch.distributed.ProcessGroup,
        intra_node_data_parallel_process_group: torch.distributed.ProcessGroup | None = None,
        model_parallel_process_group: torch.distributed.ProcessGroup | None = None,
        all_reduce_grads_across_model_parallel_group=False,
        is_hpz_layer=False,
        hpz_all_gather_stream=None,
    ):
        assert len(params) > 0
        assert all(isinstance(param, MetaParam) for param in params)

        self.name = name

        self.comm_stream = comm_stream

        self._data_parallel_process_group = data_parallel_process_group
        self.rank = data_parallel_process_group.rank()
        self.world_size = data_parallel_process_group.size()

        self.params_shapes = [p.shape for p in params]
        self.params: list[MetaParam] = params

        self.total_numel = sum([p.numel for p in self.params])
        self.padded_numel = self.total_numel

        self.is_hpz_layer = is_hpz_layer
        if self.is_hpz_layer:
            assert intra_node_data_parallel_process_group is not None
            self.hpz_rank = intra_node_data_parallel_process_group.rank()
            self.hpz_world_size = intra_node_data_parallel_process_group.size()
            self.hpz_group = intra_node_data_parallel_process_group
            self.hpz_all_gather_stream = hpz_all_gather_stream

        if self.padded_numel % self.world_size != 0:
            self.padded_numel += self.world_size - self.padded_numel % self.world_size

        flatten_buffer = torch.cat(
            [p.param.view(-1) for p in params]
            + [
                torch.zeros(
                    self.padded_numel - self.total_numel, dtype=params[0].param.dtype, device=params[0].param.device
                )
            ]
        )

        for param in self.params:
            param.clear_param()

        torch.cuda.empty_cache()  # additional seconds on process init but no fragmentation on start

        self.shard_size = self.padded_numel // self.world_size
        self.shard = flatten_buffer.narrow(0, self.shard_size * self.rank, self.shard_size).clone()

        if self.is_hpz_layer:
            self.hpz_shard_size = self.padded_numel // self.hpz_world_size
            self.hpz_inited = False

        del flatten_buffer

        torch.cuda.empty_cache()  # additional seconds on process init but no fragmentation on start

        self.weight_reusable_buffer = None
        self.grad_reusable_buffer = None
        self.first_reduce_scatter = True  # For ZeRO-2,3

        self.bit32_acc_for_bit16_reduce_scatter = False

        self.all_reduce_grads_across_model_parallel_group = all_reduce_grads_across_model_parallel_group
        if all_reduce_grads_across_model_parallel_group:
            assert model_parallel_process_group is not None
            self._model_parallel_process_group = model_parallel_process_group

    def set_buffers(self, weight_reusable_buffer, grad_reusable_buffer):
        assert self.weight_reusable_buffer is None, "attempt to set buffer for initialized supertensor"

        self.weight_reusable_buffer = weight_reusable_buffer.get_prefix(
            self.total_numel, self.padded_numel, f"{self.name}_weight"
        )
        self.grad_reusable_buffer = grad_reusable_buffer.get_prefix(
            self.total_numel, self.padded_numel, f"{self.name}_grad"
        )
        self.grad_reusable_buffer.register_free_hook(lambda stream: self.reduce_scatter_grad(stream))

    def set_fp32_grad_buffer(self, reusable_fp32_grad_buffer):
        if reusable_fp32_grad_buffer is not None:
            self.reusable_fp32_grad_buffer = reusable_fp32_grad_buffer.get_prefix(
                self.total_numel, self.padded_numel, f"{self.name}_fp32_buffer"
            )
        else:
            self.reusable_fp32_grad_buffer = None

    def set_weight_shard_buffers(self, weight_shard_buffer, fp32_shard_buffer, offset):
        old_shard = self.shard.data
        self.shard_offset = offset
        self.shard.data = weight_shard_buffer.narrow(0, offset, self.shard_size)
        self.shard.data.copy_(old_shard)
        fp32_shard_buffer.data[offset : offset + self.shard_size] = old_shard
        self.fp32_grad_shard = fp32_shard_buffer.grad.narrow(0, offset, self.shard_size)
        del old_shard
        if self.is_hpz_layer:
            self.hpz_buffer = torch.empty(self.hpz_shard_size, device=self.shard.device, dtype=self.shard.dtype)
            assert self.weight_reusable_buffer.get().shape[0] % self.hpz_buffer.shape[0] == 0

    def enable_bit32_acc_for_bit16_reduce_scatter(self):
        self.bit32_acc_for_bit16_reduce_scatter = True

    def all_gather(self, comm_stream):
        if self.weight_reusable_buffer.is_gained(self):
            return
        gather_from_hpz = self.is_hpz_layer and self.hpz_inited
        all_gather_stream = comm_stream if not gather_from_hpz else self.hpz_all_gather_stream
        with torch.cuda.stream(all_gather_stream):
            self.weight_reusable_buffer.set_prepare_stream(all_gather_stream)
            self.weight_reusable_buffer.start_prepare(self)
            torch.distributed.all_gather_into_tensor(
                self.weight_reusable_buffer.get(),
                self.shard if not gather_from_hpz else self.hpz_buffer,
                group=self._data_parallel_process_group if not gather_from_hpz else self.hpz_group,
                async_op=False,
            )
            self.weight_reusable_buffer.end_prepare(self)
            self.weight_reusable_buffer.unset_prepare_stream()

    def get_buffer_to_reduce_scatter(self, stream):
        if self.reusable_fp32_grad_buffer is None:
            return self.grad_reusable_buffer
        with torch.cuda.stream(stream):
            self.grad_reusable_buffer.start_processing(self)
            self.reusable_fp32_grad_buffer.start_prepare(self)
            self.reusable_fp32_grad_buffer.get().copy_(self.grad_reusable_buffer.get())
            self.reusable_fp32_grad_buffer.end_prepare(self)
            self.grad_reusable_buffer.end_processing(self)
        return self.reusable_fp32_grad_buffer

    def reduce_scatter_grad(self, comm_stream):
        with torch.cuda.stream(comm_stream):
            reduce_grad_buffer = self.get_buffer_to_reduce_scatter(comm_stream)
            reduce_grad_buffer.start_processing(self)

            # If it's the first call of reduce_scatter, fp32_grad_shard is zero-filled, so it can be used as the output
            reduce_in_master_grad = self.first_reduce_scatter and self.reusable_fp32_grad_buffer is not None

            if reduce_in_master_grad:
                target_tensor = self.fp32_grad_shard
            else:
                target_tensor = reduce_grad_buffer.get().narrow(
                    0, self.shard_size * self._data_parallel_process_group.rank(), self.shard_size
                )
            kwargs = dict(
                group=self._data_parallel_process_group,
                async_op=False,
            )
            if self.bit32_acc_for_bit16_reduce_scatter and target_tensor.dtype == torch.bfloat16:
                # Only bfloat16 -> float32 ReduceScatter is currently supported.
                kwargs["acc_type"] = torch.float32
            torch.distributed.reduce_scatter_tensor(
                target_tensor,
                reduce_grad_buffer.get(),
                **kwargs,
            )
            if self.all_reduce_grads_across_model_parallel_group:
                torch.distributed.all_reduce(target_tensor, group=self._model_parallel_process_group)
            if not reduce_in_master_grad:
                self.fp32_grad_shard.add_(target_tensor)

            reduce_grad_buffer.end_processing(self)
            self.first_reduce_scatter = False
            for param in self.params:
                param.grad_accumulation = False

    def init_weight_grad_flow(self):
        weight_reusable_buffer_ = self.weight_reusable_buffer.get()
        self.weight_grad_flow = torch.empty(
            0,
            dtype=weight_reusable_buffer_.dtype,
            device=weight_reusable_buffer_.device,
            requires_grad=True,
        )
        self.weight_grad_flow.data = weight_reusable_buffer_
        self.weight_grad_flow.grad = self.grad_reusable_buffer.get()

    def get_forward_pre_hook(self):
        def forward_pre_hook(module, input):
            self.all_gather(self.comm_stream)
            self.weight_reusable_buffer.start_processing(self)
            if self.is_hpz_layer and not self.hpz_inited:
                self.hpz_buffer.copy_(
                    self.weight_reusable_buffer.get().narrow(
                        0, self.hpz_shard_size * self.hpz_rank, self.hpz_shard_size
                    )
                )
                self.hpz_inited = True

        return forward_pre_hook

    def get_forward_hook(self):
        def forward_hook(module, input, output):
            self.weight_reusable_buffer.end_processing(self)

        return forward_hook

    def get_backward_pre_hook(self):
        def backward_pre_hook(module, input_grads):
            grad_buffer_is_same = self.grad_reusable_buffer.is_gained(self)
            self.grad_reusable_buffer.start_prepare(self)
            self.all_gather(self.comm_stream)
            if not grad_buffer_is_same and not self.grad_reusable_buffer.is_super_tensor_superset_of_previous_owner():
                self.grad_reusable_buffer.get().zero_()
            self.weight_reusable_buffer.start_processing(self)

        return backward_pre_hook

    def get_backward_hook(self):
        def backward_hook():
            self.weight_reusable_buffer.end_processing(self)
            self.grad_reusable_buffer.end_prepare(self)

        return backward_hook

    def pre_forward(self, inputs):
        gated_ = GateGradFlow.apply(self.get_backward_hook(), self.weight_grad_flow, *inputs)
        weight_reusable_buffer = gated_[0]
        materialize_params(self.params, weight_reusable_buffer, self.grad_reusable_buffer.get())
        return list(gated_)[1:]

    def collect_module(self):
        # build this module's parameters
        self.all_gather(self.comm_stream)  # collecting
        self.weight_reusable_buffer.start_processing(self)
        materialize_params(
            self.params, self.weight_reusable_buffer.get(), self.grad_reusable_buffer.get()
        )  # don't need grads here => None
        self.weight_reusable_buffer.end_processing(self)


class SuperTensorModule(torch.nn.Module):
    def __init__(
        self,
        root_module,
        comm_stream,
        name,
        layer_norm_module_cls: Type[nn.Module],
        device: torch.device,
        data_parallel_process_group: dist.ProcessGroup | None = None,
        intra_node_data_parallel_process_group: dist.ProcessGroup | None = None,
        model_parallel_process_group: dist.ProcessGroup | None = None,
        is_hpz_layer=False,
        hpz_all_gather_stream=None,
        sync_module_states: bool = False,
        param_init_fn: Callable | None = None,
    ):
        super(SuperTensorModule, self).__init__()
        self.name = name
        self.root_module = root_module

        is_meta_module, _ = _need_to_materialize_module(root_module, set(), set())
        if sync_module_states and is_meta_module:
            _materialize_with_param_init_fn(
                root_module=root_module,
                param_init_fn=param_init_fn,
                ignored_modules=set(),
            )
        _move_module_to_device(
            module=root_module,
            ignored_params=set(),
            ignored_buffers=set(),
            device_from_device_id=device,
        )
        if sync_module_states:
            _sync_module_states(
                params=list(root_module.parameters()),
                buffers=list(root_module.buffers()),
                process_group=intra_node_data_parallel_process_group,
            )

        # Meta
        ignored_modules = list(filter(lambda m: isinstance(m, layer_norm_module_cls), root_module.modules()))
        named_parameters_to_flatten = [
            (param_name, param)
            for module_name, module in root_module.named_modules()
            if module not in ignored_modules
            for param_name, param in module.named_parameters(module_name, recurse=False)
        ]
        params_to_flatten = [param for _, param in named_parameters_to_flatten]
        self.module_internal_fqns = [param_name for param_name, _ in named_parameters_to_flatten]
        self.module_internal_numels = [param.numel() for _, param in named_parameters_to_flatten]

        params_to_flatten = convert_some_params_to_metaparams(
            root_module, self.name, lambda param: any(param is p for p in params_to_flatten)
        )

        self.super_tensor = SuperTensor(
            params_to_flatten,
            comm_stream,
            name,
            data_parallel_process_group=data_parallel_process_group,
            intra_node_data_parallel_process_group=intra_node_data_parallel_process_group,
            model_parallel_process_group=model_parallel_process_group,
            is_hpz_layer=is_hpz_layer,
            hpz_all_gather_stream=hpz_all_gather_stream,
        )

        self.register_forward_pre_hook(self.super_tensor.get_forward_pre_hook())
        self.register_forward_hook(self.super_tensor.get_forward_hook())

        self.register_full_backward_pre_hook(self.super_tensor.get_backward_pre_hook())

    def pre_forward(self, args, kwargs):
        all_inputs = list(args) + [kwargs[key] for key in kwargs]
        gate_result = self.super_tensor.pre_forward(all_inputs)
        args = tuple(gate_result[0 : len(args)])
        for key, gated_input in zip(kwargs, args[len(args) :]):
            kwargs[key] = gated_input
        return args, kwargs

    def forward(self, *args, **kwargs):
        args, kwargs = self.pre_forward(args, kwargs)
        return self.root_module.forward(*args, **kwargs)

    def adopt_metaparams(self, main_param):
        flatten_buffer = torch.cat(
            [p.buffer for p in self.super_tensor.params]
            + [
                torch.zeros(
                    self.super_tensor.padded_numel - self.super_tensor.total_numel,
                    dtype=self.super_tensor.params[0].buffer.dtype,
                    device=self.super_tensor.params[0].buffer.device,
                )
            ]
        )
        shard = flatten_buffer.narrow(
            0, self.super_tensor.shard_size * self.super_tensor.rank, self.super_tensor.shard_size
        )  # .clone()
        with torch.no_grad():
            main_param[
                self.super_tensor.shard_offset : self.super_tensor.shard_offset + self.super_tensor.shard_size
            ].copy_(shard)
