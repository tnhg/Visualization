from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Callable, Dict, Iterable, Optional

import torch
from torch import nn

from .tensor_adapter import first_tensor, replace_first_tensor


class HookManager(AbstractContextManager):
    def __init__(self, model: nn.Module):
        self.model = model
        self.handles: list[Any] = []

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __exit__(self, *args):
        self.close()
        return False


class CaptureHookManager(HookManager):
    def __init__(self, model: nn.Module, module_names: Iterable[str], detach: bool = True, cpu: bool = False):
        super().__init__(model)
        self.outputs: Dict[str, torch.Tensor] = {}
        modules = dict(model.named_modules())
        missing = [name for name in module_names if name not in modules]
        if missing:
            raise KeyError(f"hook modules not found: {missing}")
        for name in module_names:
            self.handles.append(modules[name].register_forward_hook(self._make_hook(name, detach, cpu)))

    def _make_hook(self, name: str, detach: bool, cpu: bool):
        def hook(_module, _inputs, output):
            tensor = first_tensor(output)
            if tensor is None:
                return
            if detach:
                tensor = tensor.detach()
            if cpu:
                tensor = tensor.cpu()
            self.outputs[name] = tensor
        return hook


class InterventionHook(HookManager):
    def __init__(self, model: nn.Module, module_name: str, function: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__(model)
        modules = dict(model.named_modules())
        if module_name not in modules:
            raise KeyError(f"intervention module not found: {module_name}")

        def hook(_module, _inputs, output):
            tensor = first_tensor(output)
            if tensor is None:
                raise TypeError(f"output from {module_name} contains no tensor")
            return replace_first_tensor(output, function(tensor))

        self.handles.append(modules[module_name].register_forward_hook(hook))
