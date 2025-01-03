from abc import ABC, abstractmethod
from typing import override
import numpy as np

from .tensor import Tensor


class Criterion(ABC):
    @abstractmethod
    def __call__(self, pred:Tensor, target:Tensor) -> Tensor:...


class MeanSquaredErrorLoss(Criterion):
    @override
    def __call__(self, pred:Tensor, target:Tensor) -> Tensor:
        return (pred - target).square().mean()


class CrossEntropyLoss(Criterion):
    @override
    def __call__(self, pred:Tensor, target:Tensor) -> Tensor:
        np.clip(pred.data, 1e-38, None, pred.data)
        return -(target * pred.log()).mean()