from typing import override
import numpy as np

from .tensor import Tensor
from .module import Module, Parameter


class Linear(Module):
    def __init__(self, in_features:int, out_features:int) -> None:
        xavier = np.random.rand(in_features, out_features) 
        xavier *= np.sqrt(1 / xavier.size)
        self.w = Parameter(xavier.astype(Tensor.dtype))

    @override
    def __call__(self, a:Tensor) -> Tensor:
        return a @ self.w


class Bias1D(Module):
    def __init__(self, out_features:int) -> None:
        self.b = Parameter(np.zeros((1, out_features), Tensor.dtype))

    @override
    def __call__(self, z:Tensor) -> Tensor:
        return z + self.b
