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


class BatchNorm1D(Module):
    def __init__(self, out_features:int, epsilon:float=1e-06, momentum:float=0.1) -> None:
        self.g = Parameter(np.ones((1, out_features), Tensor.dtype))
        self.epsilon      = epsilon
        self.momentum     = momentum
        self.running_mean = Tensor(np.zeros_like(self.g, Tensor.dtype), False)
        self.running_var  = Tensor(np.zeros_like(self.g, Tensor.dtype), False)

    @override
    def __call__(self, z:Tensor) -> Tensor:
        if self._training:
            m = z.mean(axes=0, keepdims=True)
            d = z - m
            v = d.square().mean(axes=0, keepdims=True)
            with Tensor.no_grad:
                self.running_mean = (1 - self.momentum) * self.running_mean +\
                                    self.momentum * m
                self.running_var = (1 - self.momentum) * self.running_var +\
                                self.momentum * v
        else:
            d = z - self.running_mean
            v = self.running_var
        v = v + self.epsilon
        s = 1 / v.sqrt()
        x = d * s
        return self.g * x