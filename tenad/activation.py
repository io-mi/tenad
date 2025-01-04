from typing import override

from .tensor import Tensor
from .module import Module


class ReLU(Module):
    @override
    def __call__(self, z:Tensor) -> Tensor:
        return z.maximum(0)


class Tanh(Module):
    @override
    def __call__(self, z:Tensor) -> Tensor:
        return z.tanh()


class Softmax(Module):
    @override
    def __call__(self, z:Tensor) -> Tensor:
        e = (z - z.max(axes=1, keepdims=True)).exp()
        return e / e.sum(axes=1, keepdims=True)