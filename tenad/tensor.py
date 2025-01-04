from __future__ import annotations
from typing import Optional, override
from abc import ABC, abstractmethod
from numpy.typing import NDArray, DTypeLike
import numpy as np


type Operand = int | float | Tensor
type Axes = int | tuple[int, ...]
type Shape = Axes


def _broadcasted_axes(og:NDArray, bc:NDArray) -> tuple[int,...]:
    #   Broadcasting in NumPy:
    #   - Starts from the trailing dimension and works left.
    #   - Dimensions are compatible if equal or one is 1.
    #   - Resulting array has max dimensions among inputs.
    #   - Missing dimensions are treated as size 1.
    return tuple(i for i in range(-bc.ndim, 0)
        if og.ndim + i < 0 or og.shape[i] != bc.shape[i]
    )


def _broadcast_grad(data:NDArray, grad:NDArray) -> NDArray:
    return np.sum(grad, axis=axes, keepdims=True)\
    if (axes := _broadcasted_axes(data, grad)) else grad


def _argmax(arr:NDArray, axes:Axes) -> NDArray:
    if isinstance(axes, int): axes = (axes,) 
    rest = tuple(set(range(arr.ndim)).difference(axes))
    n = len(axes)
    a = np.transpose(arr, (*rest, *axes))
    b = a.reshape(*a.shape[:-n], -1)
    i = np.argmax(b, axis=-1)
    ids = np.empty((arr.ndim, *a.shape[:-n]), dtype=i.dtype)
    ids[axes,:] = np.unravel_index(i, a.shape[-n:])
    ids[rest,:] = np.indices(a.shape[:-n])
    return ids


class Operator(ABC):
    @abstractmethod
    def backprop(self, cotan:NDArray) -> None:...
    

class NoOp(Operator):
    @override
    def backprop(self, _:NDArray) -> None:
        return


class UnaryOp(Operator):
    def __init__(self, operand:Operand) -> None:
        if isinstance(operand, int) or isinstance(operand, float):
            operand = Tensor(np.array([operand], Tensor.dtype), False)
        self.operand = operand


class BinaryOp(Operator):
    def __init__(self, operand_l:Operand, operand_r:Operand) -> None:
        if isinstance(operand_l, int) or isinstance(operand_l, float):
            operand_l = Tensor(np.array([operand_l], Tensor.dtype), False)
        if isinstance(operand_r, int) or isinstance(operand_r, float):
            operand_r = Tensor(np.array([operand_r], Tensor.dtype), False)
        self.operand_l = operand_l
        self.operand_r = operand_r


class NegOp(UnaryOp):
    def __init__(self, operand:Operand) -> None:
        UnaryOp.__init__(self, operand)
        self.result = Tensor(-self.operand.data, not Tensor.no_grad.enabled, self)
    
    @override
    def backprop(self, cotan:NDArray) -> None:
        self.operand.grad -= cotan
        self.operand.op.backprop(-cotan)


class AddOp(BinaryOp):
    def __init__(self, operand_l:Operand, operand_r:Operand) -> None:
        BinaryOp.__init__(self, operand_l, operand_r)
        self.result = Tensor(self.operand_l.data + self.operand_r.data, 
                             not Tensor.no_grad.enabled, self)

    @override
    def backprop(self, cotan:NDArray) -> None:
        if self.operand_l.grad.size != 0:
            grad = _broadcast_grad(self.operand_l.data, cotan)
            self.operand_l.grad += grad
            self.operand_l.op.backprop(grad)
        if self.operand_r.grad.size != 0:
            grad = _broadcast_grad(self.operand_r.data, cotan)
            self.operand_r.grad += grad
            self.operand_r.op.backprop(grad)


class SubOp(BinaryOp):
    def __init__(self, operand_l:Operand, operand_r:Operand) -> None:
        BinaryOp.__init__(self, operand_l, operand_r)
        self.result = Tensor(self.operand_l.data - self.operand_r.data, 
                             not Tensor.no_grad.enabled, self)

    @override
    def backprop(self, cotan:NDArray) -> None:
        if self.operand_l.grad.size != 0:
            grad = _broadcast_grad(self.operand_l.data, cotan)
            self.operand_l.grad += grad
            self.operand_l.op.backprop(grad)
        if self.operand_r.grad.size != 0:
            grad = _broadcast_grad(self.operand_r.data,-cotan)
            self.operand_r.grad += grad
            self.operand_r.op.backprop(grad)


class MulOp(BinaryOp):
    def __init__(self, operand_l:Operand, operand_r:Operand) -> None:
        BinaryOp.__init__(self, operand_l, operand_r)
        self.result = Tensor(self.operand_l.data * self.operand_r.data, 
                             not Tensor.no_grad.enabled, self)

    @override
    def backprop(self, cotan:NDArray) -> None:
        if self.operand_l.grad.size != 0:
            grad = _broadcast_grad(self.operand_l.data, cotan*self.operand_r.data)
            self.operand_l.grad += grad
            self.operand_l.op.backprop(grad)
        if self.operand_r.grad.size != 0:
            grad = _broadcast_grad(self.operand_r.data, cotan*self.operand_l.data)
            self.operand_r.grad += grad
            self.operand_r.op.backprop(grad)


class DivOp(BinaryOp):
    def __init__(self, operand_l:Operand, operand_r:Operand) -> None:
        BinaryOp.__init__(self, operand_l, operand_r)
        self.result = Tensor(self.operand_l.data / self.operand_r.data, 
                             not Tensor.no_grad.enabled, self)

    @override
    def backprop(self, cotan:NDArray) -> None:
        rcp = 1 / self.operand_r.data
        if self.operand_l.grad.size != 0:
            grad = _broadcast_grad(self.operand_l.data, cotan*rcp)
            self.operand_l.grad += grad
            self.operand_l.op.backprop(grad)
        if self.operand_r.grad.size != 0:
            grad = _broadcast_grad(self.operand_r.data, 
                        cotan * -(self.operand_l.data*rcp**2))
            self.operand_r.grad += grad
            self.operand_r.op.backprop(grad)


class MatMulOp(BinaryOp):
    def __init__(self, operand_l:Operand, operand_r:Operand) -> None:
        BinaryOp.__init__(self, operand_l, operand_r)
        self.result = Tensor(self.operand_l.data @ self.operand_r.data, 
                             not Tensor.no_grad.enabled, self)

    @override
    def backprop(self, cotan:NDArray) -> None:
        if self.operand_l.grad.size != 0:
            #   TODO: same as self.operand_r.data @ cotan ?
            grad = cotan @ self.operand_r.data.T
            self.operand_l.grad += grad
            self.operand_l.op.backprop(grad)
        if self.operand_r.grad.size != 0:
            #   TODO: same as cotan @ self.operand_l.data ?
            grad = self.operand_l.data.T @ cotan
            self.operand_r.grad += grad
            self.operand_r.op.backprop(grad)


class PowOp(BinaryOp):
    def __init__(self, operand_l:Operand, operand_r:Operand) -> None:
        BinaryOp.__init__(self, operand_l, operand_r)
        self.result = Tensor(self.operand_l.data ** self.operand_r.data, 
                             not Tensor.no_grad.enabled, self)

    @override
    def backprop(self, cotan:NDArray) -> None:
        if self.operand_l.grad.size != 0:
            grad = _broadcast_grad(self.operand_l.data, 
                        cotan * self.operand_r.data * self.operand_l.data ** (self.operand_r.data-1.0))
            self.operand_l.grad += grad
            self.operand_l.op.backprop(grad)
        if self.operand_r.grad.size != 0:
            grad = _broadcast_grad(self.operand_r.data,
                    cotan * np.log(self.operand_l.data) * self.result.data)
            self.operand_r.grad += grad
            self.operand_r.op.backprop(grad)


class MaximumOp(BinaryOp):
    def __init__(self, operand_l:Operand, operand_r:Operand) -> None:
        BinaryOp.__init__(self, operand_l, operand_r)
        self.result = Tensor(np.maximum(self.operand_l.data, self.operand_r.data), 
                             not Tensor.no_grad.enabled, self)

    @override
    def backprop(self, cotan:NDArray) -> None:
        mask = np.equal(self.result.data, self.operand_l.data)
        if self.operand_l.grad.size != 0:
            grad = _broadcast_grad(self.operand_l.data, cotan * mask)
            self.operand_l.grad += grad
            self.operand_l.op.backprop(grad)
        if self.operand_r.grad.size != 0:
            grad = _broadcast_grad(self.operand_r.data, cotan * np.logical_not(mask))
            self.operand_r.grad += grad
            self.operand_r.op.backprop(grad)


class SquareOp(UnaryOp):
    def __init__(self, operand:Operand) -> None:
        UnaryOp.__init__(self, operand)
        self.result = Tensor(np.square(self.operand.data), not Tensor.no_grad.enabled, self)

    @override
    def backprop(self, cotan:NDArray) -> None:
        if self.operand.grad.size == 0: return
        grad = cotan * 2 * self.operand.data
        self.operand.grad += grad
        self.operand.op.backprop(grad)


class LogOp(UnaryOp):
    def __init__(self, operand:Operand) -> None:
        UnaryOp.__init__(self, operand)
        self.result = Tensor(np.log(self.operand.data), not Tensor.no_grad.enabled, self)

    @override
    def backprop(self, cotan:NDArray) -> None:
        if self.operand.grad.size == 0: return
        grad = cotan / self.operand.data
        self.operand.grad += grad
        self.operand.op.backprop(grad)


class ExpOp(UnaryOp):
    def __init__(self, operand:Operand) -> None:
        UnaryOp.__init__(self, operand)
        self.result = Tensor(np.exp(self.operand.data), not Tensor.no_grad.enabled, self)

    @override
    def backprop(self, cotan:NDArray) -> None:
        if self.operand.grad.size == 0: return
        grad = cotan * self.result.data
        self.operand.grad += grad
        self.operand.op.backprop(grad)


class SqrtOp(UnaryOp):
    def __init__(self, operand:Operand) -> None:
        UnaryOp.__init__(self, operand)
        self.result = Tensor(np.sqrt(self.operand.data), not Tensor.no_grad.enabled, self)

    @override
    def backprop(self, cotan:NDArray) -> None:
        if self.operand.grad.size == 0: return
        grad = cotan * 1 / (2 * self.result.data)
        self.operand.grad += grad
        self.operand.op.backprop(grad)


class TanhOp(UnaryOp):
    def __init__(self, operand:Operand) -> None:
        UnaryOp.__init__(self, operand)
        self.result = Tensor(np.tanh(self.operand.data), not Tensor.no_grad.enabled, self)

    @override
    def backprop(self, cotan:NDArray) -> None:
        if self.operand.grad.size == 0: return
        grad = cotan * (1-self.result.data**2)
        self.operand.grad += grad
        self.operand.op.backprop(grad)


class SumOp(UnaryOp):
    def __init__(self, operand:Operand, axes:Optional[Axes], keepdims:bool) -> None:
        UnaryOp.__init__(self, operand)
        self.result = Tensor(
            np.atleast_1d(np.sum(self.operand.data, axis=axes, keepdims=keepdims)),
            not Tensor.no_grad.enabled, self
        )

    @override
    def backprop(self, cotan:NDArray) -> None:
        if self.operand.grad.size == 0: return
        grad = cotan * np.ones_like(self.operand.data)
        self.operand.grad += grad
        self.operand.op.backprop(grad)


class AbsOp(UnaryOp):
    def __init__(self, operand:Operand) -> None:
        UnaryOp.__init__(self, operand)
        self.result = Tensor(np.abs(self.operand.data), not Tensor.no_grad.enabled, self)

    @override
    def backprop(self, cotan:NDArray) -> None:
        if self.operand.grad.size == 0: return
        grad = cotan * np.sign(self.operand.data)
        self.operand.grad += grad
        self.operand.op.backprop(grad)


class MaxOp(UnaryOp):
    def __init__(self, operand:Operand, axes:Axes, keepdims:bool) -> None:
        UnaryOp.__init__(self, operand)
        self._indices = _argmax(self.operand.data, axes)
        values = self.operand.data[*self._indices]
        if keepdims: values = np.expand_dims(values, axes)
        self.result = Tensor(values, not Tensor.no_grad.enabled, self)

    @override
    def backprop(self, cotan:NDArray) -> None:
        if self.operand.grad.size == 0: return
        grad = np.zeros_like(self.operand.data)
        grad[*self._indices] = 1.0
        grad *= cotan
        self.operand.grad += grad
        self.operand.op.backprop(grad)


class MeanOp(UnaryOp):
    def __init__(self, operand:Operand, axes:Optional[Axes], keepdims:bool) -> None:
        UnaryOp.__init__(self, operand)
        self._axes = axes
        self.result = Tensor(
            np.atleast_1d(np.mean(self.operand.data, axis=axes, keepdims=keepdims)), 
            not Tensor.no_grad.enabled, self
        )

    @override
    def backprop(self, cotan:NDArray) -> None:
        if self.operand.grad.size == 0: return
        denom = self.operand.data.size
        if isinstance(self._axes, int): 
            denom = self.operand.data.shape[self._axes]
        elif isinstance(self._axes, tuple):
            denom = np.prod(tuple(self.operand.data.shape[i] for i in self._axes))
        grad = cotan * np.ones_like(self.operand.data) / denom
        self.operand.grad += grad
        self.operand.op.backprop(grad)


class TensorDataType:
    def __init__(self, dtype:DTypeLike) -> None:
        self.dtype = dtype 

    def __get__(self, obj, objtype=None) -> DTypeLike:
        return self.dtype

    def __set__(self, obj, value:DTypeLike) -> None:
        assert np.issubdtype(value, np.floating)
        self.dtype = value


class TensorNoGrad:
    def __init__(self) -> None:
        self.enabled = False

    def __enter__(self) -> None:
        self.enabled = True

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        self.enabled = False

    def __bool__(self) -> bool:
        return self.enabled


# By defining attributes in the metaclass, you can control accessibility and 
# ensure that certain attributes are accessible only through the class itself and 
# not through its instances.
class TensorMeta(type):
    dtype = TensorDataType(np.float32)
    no_grad = TensorNoGrad()


class Tensor(metaclass=TensorMeta):
    def __init__(self, data:NDArray, use_grad:bool=True, op:Operator=NoOp()) -> None:
        assert data.dtype == Tensor.dtype
        self.data = data
        self.grad = np.zeros_like(data) if use_grad else np.empty(0)
        self.op   = op

    @staticmethod
    def zeros(shape:Shape, use_grad:bool=True) -> Tensor:
        return Tensor(np.zeros(shape, Tensor.dtype), use_grad)

    @staticmethod
    def ones(shape:Shape, use_grad:bool=True) -> Tensor:
        return Tensor(np.ones(shape, Tensor.dtype), use_grad)

    @staticmethod
    def one_hot(classes:NDArray, num_classes:int=-1) -> Tensor:
        if num_classes < 1:
            num_classes = classes.max() + 1
        idx = np.expand_dims(classes, axis=-1)
        encoded = np.zeros(classes.shape + (num_classes,), dtype=Tensor.dtype)
        np.put_along_axis(encoded, idx, 1.0, axis=-1)
        return Tensor(encoded, False)

    def backprop(self) -> None:
        self.op.backprop(np.ones_like(self.data))

    def __neg__(self) -> Tensor:
        op = NegOp(self)
        return op.result

    def __add__(self, other:Operand) -> Tensor:
        op = AddOp(self, other)
        return op.result

    def __radd__(self, other:Operand) -> Tensor:
        op = AddOp(other, self)
        return op.result

    def __sub__(self, other:Operand) -> Tensor:
        op = SubOp(self, other)
        return op.result

    def __rsub__(self, other:Operand) -> Tensor:
        op = SubOp(other, self)
        return op.result

    def __mul__(self, other:Operand) -> Tensor:
        op = MulOp(self, other)
        return op.result

    def __rmul__(self, other:Operand) -> Tensor:
        op = MulOp(other, self)
        return op.result

    def __truediv__(self, other:Operand) -> Tensor:
        op = DivOp(self, other)
        return op.result

    def __rtruediv__(self, other:Operand) -> Tensor:
        op = DivOp(other, self)
        return op.result

    def __pow__(self, other:Operand) -> Tensor:
        op = PowOp(self, other)
        return op.result

    def __matmul__(self, other:Operand) -> Tensor:
        op = MatMulOp(self, other)
        return op.result

    def square(self) -> Tensor:
        op = SquareOp(self)
        return op.result

    def log(self) -> Tensor:
        op = LogOp(self)
        return op.result

    def exp(self) -> Tensor:
        op = ExpOp(self)
        return op.result

    def sqrt(self) -> Tensor:
        op = SqrtOp(self)
        return op.result

    def tanh(self) -> Tensor:
        op = TanhOp(self)
        return op.result

    def sum(self, axes:Optional[Axes] = None, keepdims:bool = False) -> Tensor:
        op = SumOp(self, axes, keepdims)
        return op.result

    def mean(self, axes:Optional[Axes] = None, keepdims:bool = False) -> Tensor:
        op = MeanOp(self, axes, keepdims)
        return op.result

    def abs(self) -> Tensor:
        op = AbsOp(self)
        return op.result

    def maximum(self, t:Operand) -> Tensor:
        op = MaximumOp(self, t)
        return op.result

    def max(self, axes:Axes, keepdims:bool = False) -> Tensor:
        op = MaxOp(self, axes, keepdims)
        return op.result
