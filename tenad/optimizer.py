from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import override
import numpy as np

from .module import Parameter


class Optimizer(ABC):
    def __init__(self, parameters:Iterator[Parameter], learning_rate:float, 
                       minimize:bool=True) -> None:
        self._parameters = tuple(parameters)
        self._rate = -learning_rate if minimize else learning_rate
        self._epoch_count = 0

    @abstractmethod
    def step(self) -> None:
        self._epoch_count += 1


#   SGD
class StochasticGradientDescent(Optimizer):
    def __init__(self, parameters:Iterator[Parameter], 
                       learning_rate:float, minimize:bool=True, 
                       moment:float=0.9) -> None:
        Optimizer.__init__(self, parameters, learning_rate, minimize)
        self._moment = moment
        self._grad_exp_avg = tuple(
            np.zeros_like(param.data) for param in self._parameters
        )

    @override
    def step(self) -> None:
        Optimizer.step(self)
        for i, param in enumerate(self._parameters):
            grad_exp_avg = self._grad_exp_avg[i]
            grad_exp_avg *= self._moment
            grad_exp_avg += (1 - self._moment) * param.grad
            param.data += self._rate * grad_exp_avg
