from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import torch
import torch.nn as nn

from ._common import TrainingState


class ShardedState(Enum):
    """
    - ``SHARDED``: The sharded parameter is registered to the module. It is the
      only contributor to parameter memory.
    - ``SHARDED_POST_FORWARD``: The unsharded parameter is resharded to a
      smaller world size. Since this data should not be used for computation,
      we do not register it to the module. Users should reshard the module
      before any in-place modifications. Both it and the sharded parameter
      contribute to parameter memory.
    - ``UNSHARDED``: The unsharded parameter is registered to the module. Both
      it and the sharded parameter contribute to parameter memory.
    """

    SHARDED = auto()
    SHARDED_POST_FORWARD = auto()
    UNSHARDED = auto()


@dataclass
class ParamModuleInfo:
    """
    For a parameter, this stores the module and the parameter name to be able
    to do a parameter swap via ``setattr(module, param_name, ...)`` or to get
    the parameter via ``getattr(module, param_name)``. We additionally save
    shared modules and shared parameter names to update them accordingly.
    """

    # Parameter names are unprefixed, e.g. "weight", not "lin.weight"
    module: nn.Module
    param_name: str
    shared_modules: list[nn.Module] = field(default_factory=list)
    shared_param_names: list[str] = field(default_factory=list)


class YaFSDPParam:
    """
    This class manages a parameter with YaFSDP applied.
    """

    _orig_size: torch.Size
    _orig_numel: int
    sharded_param: nn.Parameter
    _sharded_param_grad: torch.Tensor
    _unsharded_param: nn.Parameter
    _unsharded_param_grad: torch.Tensor
    _unsharded_params_for_backward: list[nn.Parameter]

    def __init__(
        self,
        param: nn.Parameter,
        sharded_param: nn.Parameter,
        sharded_param_grad: torch.Tensor,
        module_info: ParamModuleInfo,
        device: torch.device,
    ):
        self._module_info: ParamModuleInfo = module_info
        self.device = device
        self._init_sharded_param(param, sharded_param, sharded_param_grad, device)
        self._unsharded_param = None
        self._unsharded_param_grad = None
        self._unsharded_params_for_backward = []
        self._param_fqn: Optional[str] = None  # prefixed from root module

    @torch.no_grad()
    def _init_sharded_param(
        self,
        param: nn.Parameter,
        sharded_param: nn.Parameter,
        sharded_param_grad: torch.Tensor,
        device: torch.device,
    ):
        if param.device != device:
            raise AssertionError(f"Expects the parameter to already be moved to device {device} but got {param.device}")
        self._orig_size = param.size()
        self._orig_numel = param.numel()
        self.sharded_param = sharded_param
        self.sharded_param.requires_grad_(param.requires_grad)
        self._sharded_param_grad = sharded_param_grad
        self._setattr_on_modules(self.sharded_param)
        self.sharded_state = ShardedState.SHARDED

    def set_sharded_param_grad(self):
        self.sharded_param.grad = self._sharded_param_grad

    @torch.no_grad()
    def init_unsharded_param(
        self, padded_unsharded_data: torch.Tensor, padded_unsharded_grad: torch.Tensor, offset: int
    ):
        self._unsharded_param = nn.Parameter(
            padded_unsharded_data.narrow(0, offset, self._orig_numel).view(self._orig_size),
            requires_grad=self.sharded_param.requires_grad,
        )
        if self.sharded_param.requires_grad:
            self._unsharded_param_grad = padded_unsharded_grad.narrow(0, offset, self._orig_numel).view(self._orig_size)

    def to_sharded(self) -> None:
        self._setattr_on_modules(self.sharded_param)
        self.sharded_state = ShardedState.SHARDED

    # def to_sharded_post_forward(self) -> None:
    #     self._setattr_on_modules(self._sharded_post_forward_param)
    #     self.free_unsharded_param()
    #     self.sharded_state = ShardedState.SHARDED_POST_FORWARD

    def to_unsharded(self, training_state: Optional[TrainingState] = None) -> None:
        # Assume that the data has been allocated and all-gathered
        if training_state == TrainingState.FORWARD and torch.is_grad_enabled() and self.sharded_param.requires_grad:
            self._setattr_on_modules(self._unsharded_params_for_backward[-1])
        elif training_state == TrainingState.PRE_BACKWARD and self.sharded_param.requires_grad:
            self._setattr_on_modules(self._unsharded_params_for_backward.pop())
        else:
            self._setattr_on_modules(self._unsharded_param)
        # if self.sharded_state == ShardedState.SHARDED_POST_FORWARD:
        #     # The data is allocated in the default stream via the post-forward
        #     # reshard and must be kept alive for the next all-gather copy-in.
        #     # Since we call this method after the copy-out, the data's lifetime
        #     # is ensured without further synchronization.
        #     self._sharded_post_forward_param = None
        #     self._sharded_post_forward_param_data = None  # free
        self.sharded_state = ShardedState.UNSHARDED

    def _setattr_on_modules(self, param: nn.Parameter) -> None:
        unsafe_setattr_param(self._module_info.module, self._module_info.param_name, param)
        for shared_module, shared_param_name in zip(
            self._module_info.shared_modules, self._module_info.shared_param_names
        ):
            unsafe_setattr_param(shared_module, shared_param_name, param)


# NOTE: These bypass `nn.Module.__setattr__` checks, which incur non-trivial
# CPU overhead, if the module did not override it. For FSDP, we know we do not
# need those checks when transitioning between sharded/unsharded parameters.
def unsafe_setattr_param(module: nn.Module, param_name: str, param: nn.Parameter) -> None:
    if getattr(module.__setattr__, "__func__", None) is nn.Module.__setattr__:
        module._parameters[param_name] = param
    else:  # slow path
        setattr(module, param_name, param)
