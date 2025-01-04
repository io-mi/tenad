from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from .tensor import Tensor


class Parameter(Tensor):
    pass


class ModuleMode:
    def __init__(self, module:Module) -> None:
        self._module = module
    
    def __enter__(self) -> None:
        self._module._training = True
        for m in self._module.modules():
            m._training = True
    
    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        self._module._training = False
        for m in self._module.modules():
            m._training = False


class Module(ABC):
    #   Using __new__ to ensure that every Module subclass automatically gets 
    #   the _modules and _parameters dictionaries without requiring 
    #   super().__init__() in the subclass's __init__.
    def __new__(cls, *args, **kwargs) -> Module:
        module = object.__new__(cls)
        module._modules = dict()
        module._parameters = dict()
        module._training = True
        module.__init__(*args, **kwargs)
        return module

    def __init__(self) -> None:
        self._modules: dict[str, Module]
        self._parameters: dict[str, Parameter]
        self._training: bool

    @abstractmethod
    def __call__(self, x:Tensor) -> Tensor:...

    def __setattr__(self, name:str, value:Any) -> None:
        if isinstance(value, Parameter): self._parameters[name] = value
        elif isinstance(value, Module): self._modules[name] = value
        object.__setattr__(self, name, value)

    def zero_grad(self) -> None:
        for param in self.parameters():
            param.grad[:] = 0

    def modules(self) -> Iterator[Module]:
        for module in self._modules.values():
            yield module

    def parameters(self) -> Iterator[Parameter]:
        for module in self.modules():
            yield from module.parameters()
        yield from self._parameters.values()

    @property
    def evaluate(self) -> ModuleMode:
        return ModuleMode(self)
