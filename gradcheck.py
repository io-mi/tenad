from __future__ import annotations
from numpy.typing import NDArray
import numpy as np
from tensor import Tensor


def wrap(array:NDArray, *indices:int) -> tuple[int, ...]:
    return tuple(i%n for i,n in zip(indices, array.shape))


class GradCheck:
    def __init__(self, delta:float = 1e-7, tolerance: float = 1e-5) -> None:
        # TODO: tolerance from delta
        self.delta = delta
        self.tolerance = tolerance

    def __enter__(self) -> GradCheck:
        self.prev_dtype = Tensor.dtype
        Tensor.dtype = np.float64
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        Tensor.dtype = self.prev_dtype

    def log_entry(self, name:str, error:float) -> str:
        return "{0:<9} {1}".format(name, error)
    
    def abs(self) -> str:
        v = Tensor(np.random.uniform(-1.0, 1.0, (36, 24)).astype(np.float64))
        v.abs().backprop()  
        v_data = v.data.copy()
        v_grad = np.zeros_like(v.data)
        for i in range(v.data.shape[0]):
            for j in range(v.data.shape[1]):
                v_data[i,j] -= self.delta
                v0 = np.abs(v_data[i,j])
                v_data[i,j] = v.data[i,j]
                v_data[i,j] += self.delta
                v1 = np.abs(v_data[i,j])
                v_data[i,j] = v.data[i,j]
                v_grad[i,j] += (v1 - v0) / (2.0 * self.delta)
        return self.log_entry("abs", 
                    np.sqrt(np.sum(np.power(v.grad - v_grad, 2.0))) / 
                    np.sqrt(np.sum(np.power(v.grad + v_grad, 2.0)))
                )

    def max(self) -> str:
        v = Tensor(np.random.rand(32, 64).astype(np.float64))
        v.max(1, keepdims=True).backprop()
        v_data = v.data.copy()
        v_grad = np.zeros_like(v.data)
        for j in range(v.data.shape[1]):
            for i in range(v.data.shape[0]):
                v_data[i,j] -= self.delta
                v0 = np.max(v_data[i,:])
                v_data[i,j] = v.data[i,j]
                v_data[i,j] += self.delta
                v1 = np.max(v_data[i,:])
                v_data[i,j] = v.data[i,j]
                v_grad[i,j] += (v1 - v0) / (2.0 * self.delta)
        return self.log_entry("max", 
                    np.sqrt(np.sum(np.power(v.grad - v_grad, 2.0))) / 
                    np.sqrt(np.sum(np.power(v.grad + v_grad, 2.0)))
                )
                    
    def neg(self) -> str:
        v = Tensor(np.random.rand(40, 40).astype(np.float64))
        (-v).backprop()  
        v_data = v.data.copy()
        v_grad = np.zeros_like(v.data)
        for i in range(v.data.shape[0]):
            for j in range(v.data.shape[1]):
                v_data[i,j] -= self.delta
                v0 = -v_data[i,j]
                v_data[i,j] = v.data[i,j]
                v_data[i,j] += self.delta
                v1 = -v_data[i,j]
                v_data[i,j] = v.data[i,j]
                v_grad[i,j] += (v1 - v0) / (2.0 * self.delta)
        return self.log_entry("neg", 
                    np.sqrt(np.sum(np.power(v.grad - v_grad, 2.0))) / 
                    np.sqrt(np.sum(np.power(v.grad + v_grad, 2.0)))
                )

    def square(self) -> str:
        v = Tensor(np.random.rand(40, 40).astype(np.float64))
        v.square().backprop()  
        v_data = v.data.copy()
        v_grad = np.zeros_like(v.data)
        for i in range(v.data.shape[0]):
            for j in range(v.data.shape[1]):
                v_data[i,j] -= self.delta
                v0 = np.square(v_data[i,j])
                v_data[i,j] = v.data[i,j]
                v_data[i,j] += self.delta
                v1 = np.square(v_data[i,j])
                v_data[i,j] = v.data[i,j]
                v_grad[i,j] += (v1 - v0) / (2.0 * self.delta)
        return self.log_entry("square", 
                    np.sqrt(np.sum(np.power(v.grad - v_grad, 2.0))) / 
                    np.sqrt(np.sum(np.power(v.grad + v_grad, 2.0)))
                )

    def log(self) -> str:
        v = Tensor(np.random.rand(40, 40).astype(np.float64))
        v.log().backprop()  
        v_data = v.data.copy()
        v_grad = np.zeros_like(v.data)
        for i in range(v.data.shape[0]):
            for j in range(v.data.shape[1]):
                v_data[i,j] -= self.delta
                v0 = np.log(v_data[i,j])
                v_data[i,j] = v.data[i,j]
                v_data[i,j] += self.delta
                v1 = np.log(v_data[i,j])
                v_data[i,j] = v.data[i,j]
                v_grad[i,j] += (v1 - v0) / (2.0 * self.delta)
        return self.log_entry("log", 
                    np.sqrt(np.sum(np.power(v.grad - v_grad, 2.0))) / 
                    np.sqrt(np.sum(np.power(v.grad + v_grad, 2.0)))
                )
    
    def exp(self) -> str:
        v = Tensor(np.random.rand(40, 40).astype(np.float64))
        v.exp().backprop()  
        v_data = v.data.copy()
        v_grad = np.zeros_like(v.data)
        for i in range(v.data.shape[0]):
            for j in range(v.data.shape[1]):
                v_data[i,j] -= self.delta
                v0 = np.exp(v_data[i,j])
                v_data[i,j] = v.data[i,j]
                v_data[i,j] += self.delta
                v1 = np.exp(v_data[i,j])
                v_data[i,j] = v.data[i,j]
                v_grad[i,j] += (v1 - v0) / (2.0 * self.delta)
        return self.log_entry("exp", 
                    np.sqrt(np.sum(np.power(v.grad - v_grad, 2.0))) / 
                    np.sqrt(np.sum(np.power(v.grad + v_grad, 2.0)))
                )

    def sqrt(self) -> str:
        v = Tensor(np.random.rand(40, 40).astype(np.float64))
        v.sqrt().backprop()  
        v_data = v.data.copy()
        v_grad = np.zeros_like(v.data)
        for i in range(v.data.shape[0]):
            for j in range(v.data.shape[1]):
                v_data[i,j] -= self.delta
                v0 = np.sqrt(v_data[i,j])
                v_data[i,j] = v.data[i,j]
                v_data[i,j] += self.delta
                v1 = np.sqrt(v_data[i,j])
                v_data[i,j] = v.data[i,j]
                v_grad[i,j] += (v1 - v0) / (2.0 * self.delta)
        return self.log_entry("sqrt", 
                    np.sqrt(np.sum(np.power(v.grad - v_grad, 2.0))) / 
                    np.sqrt(np.sum(np.power(v.grad + v_grad, 2.0)))
                )
    
    def tanh(self) -> str:
        v = Tensor(np.random.rand(40, 40).astype(np.float64))
        v.tanh().backprop()  
        v_data = v.data.copy()
        v_grad = np.zeros_like(v.data)
        for i in range(v.data.shape[0]):
            for j in range(v.data.shape[1]):
                v_data[i,j] -= self.delta
                v0 = np.tanh(v_data[i,j])
                v_data[i,j] = v.data[i,j]
                v_data[i,j] += self.delta
                v1 = np.tanh(v_data[i,j])
                v_data[i,j] = v.data[i,j]
                v_grad[i,j] += (v1 - v0) / (2.0 * self.delta)
        return self.log_entry("tanh", 
                    np.sqrt(np.sum(np.power(v.grad - v_grad, 2.0))) / 
                    np.sqrt(np.sum(np.power(v.grad + v_grad, 2.0)))
                )

    def sum(self) -> str:
        v = Tensor(np.random.rand(32, 64).astype(np.float64))
        v.sum(1, keepdims=True).backprop()
        v_data = v.data.copy()
        v_grad = np.zeros_like(v.data)
        for j in range(v.data.shape[1]):
            for i in range(v.data.shape[0]):
                v_data[i,j] -= self.delta
                v0 = np.sum(v_data[i,:])
                v_data[i,j] = v.data[i,j]
                v_data[i,j] += self.delta
                v1 = np.sum(v_data[i,:])
                v_data[i,j] = v.data[i,j]
                v_grad[i,j] += (v1 - v0) / (2.0 * self.delta)
        return self.log_entry("sum", 
                    np.sqrt(np.sum(np.power(v.grad - v_grad, 2.0))) / 
                    np.sqrt(np.sum(np.power(v.grad + v_grad, 2.0)))
                )

    def mean(self) -> str:
        v = Tensor(np.random.rand(32, 64).astype(np.float64))
        v.mean(1, keepdims=True).backprop()
        v_data = v.data.copy()
        v_grad = np.zeros_like(v.data)
        for j in range(v.data.shape[1]):
            for i in range(v.data.shape[0]):
                v_data[i,j] -= self.delta
                v0 = np.mean(v_data[i,:])
                v_data[i,j] = v.data[i,j]
                v_data[i,j] += self.delta
                v1 = np.mean(v_data[i,:])
                v_data[i,j] = v.data[i,j]
                v_grad[i,j] += (v1 - v0) / (2.0 * self.delta)
        return self.log_entry("mean", 
                    np.sqrt(np.sum(np.power(v.grad - v_grad, 2.0))) / 
                    np.sqrt(np.sum(np.power(v.grad + v_grad, 2.0)))
                )

    def maximum(self) -> str:
        l = Tensor(np.random.uniform(-100, 100, (36, 48)).astype(np.float64))
        r = Tensor(np.random.uniform(-100, 100, (36, 48)).astype(np.float64))
        l.maximum(r).backprop()
        l_data = l.data.copy()
        r_data = r.data.copy()
        l_grad = np.zeros_like(l.data)
        r_grad = np.zeros_like(r.data)
        shape = np.broadcast_shapes(l.data.shape, r.data.shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                l_data[wrap(l_data,i,j)] -= self.delta
                v0 = np.maximum(l_data[wrap(l_data,i,j)], r.data[wrap(r_data,i,j)])
                l_data[wrap(l_data,i,j)] = l.data[wrap(l_data,i,j)]
                l_data[wrap(l_data,i,j)] += self.delta
                v1 = np.maximum(l_data[wrap(l_data,i,j)], r.data[wrap(r_data,i,j)])
                l_data[wrap(l_data,i,j)] = l.data[wrap(l_data,i,j)]
                l_grad[wrap(l_data,i,j)] += (v1 - v0) / (2.0 * self.delta)
                r_data[wrap(r_data,i,j)] -= self.delta
                v0 = np.maximum(l.data[wrap(l_data,i,j)], r_data[wrap(r_data,i,j)])
                r_data[wrap(r_data,i,j)] = r.data[wrap(r_data,i,j)]
                r_data[wrap(r_data,i,j)] += self.delta
                v1 = np.maximum(l.data[wrap(l_data,i,j)], r_data[wrap(r_data,i,j)])
                r_data[wrap(r_data,i,j)] = r.data[wrap(r_data,i,j)]
                r_grad[wrap(r_data,i,j)] += (v1 - v0) / (2.0 * self.delta)
        return "\n".join((
                self.log_entry("maximum.l", 
                    np.sqrt(np.sum(np.power(l.grad - l_grad, 2.0))) /
                    np.sqrt(np.sum(np.power(l.grad + l_grad, 2.0)))),
                self.log_entry("maximum.r", 
                    np.sqrt(np.sum(np.power(r.grad - r_grad, 2.0))) /
                    np.sqrt(np.sum(np.power(r.grad + r_grad, 2.0))))
                ))

    def add(self) -> str:
        l = Tensor(np.random.rand(24, 1).astype(np.float64))
        r = Tensor(np.random.rand(1, 46).astype(np.float64))
        (l + r).backprop()
        l_data = l.data.copy()
        r_data = r.data.copy()
        l_grad = np.zeros_like(l.data)
        r_grad = np.zeros_like(r.data)
        shape = np.broadcast_shapes(l.data.shape, r.data.shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                l_data[wrap(l_data,i,j)] -= self.delta
                v0 = l_data[wrap(l_data,i,j)] + r.data[wrap(r_data,i,j)]
                l_data[wrap(l_data,i,j)] = l.data[wrap(l_data,i,j)]
                l_data[wrap(l_data,i,j)] += self.delta
                v1 = l_data[wrap(l_data,i,j)] + r.data[wrap(r_data,i,j)]
                l_data[wrap(l_data,i,j)] = l.data[wrap(l_data,i,j)]
                l_grad[wrap(l_data,i,j)] += (v1 - v0) / (2.0 * self.delta)
                r_data[wrap(r_data,i,j)] -= self.delta
                v0 = l.data[wrap(l_data,i,j)] + r_data[wrap(r_data,i,j)]
                r_data[wrap(r_data,i,j)] = r.data[wrap(r_data,i,j)]
                r_data[wrap(r_data,i,j)] += self.delta
                v1 = l.data[wrap(l_data,i,j)] + r_data[wrap(r_data,i,j)]
                r_data[wrap(r_data,i,j)] = r.data[wrap(r_data,i,j)]
                r_grad[wrap(r_data,i,j)] += (v1 - v0) / (2.0 * self.delta)
        return "\n".join((
                self.log_entry("add.l", 
                    np.sqrt(np.sum(np.power(l.grad - l_grad, 2.0))) /
                    np.sqrt(np.sum(np.power(l.grad + l_grad, 2.0)))),
                self.log_entry("add.r", 
                    np.sqrt(np.sum(np.power(r.grad - r_grad, 2.0))) /
                    np.sqrt(np.sum(np.power(r.grad + r_grad, 2.0))))
                ))

    def sub(self) -> str:
        l = Tensor(np.random.rand(24, 1).astype(np.float64))
        r = Tensor(np.random.rand(1, 46).astype(np.float64))
        (l - r).backprop()
        l_data = l.data.copy()
        r_data = r.data.copy()
        l_grad = np.zeros_like(l.data)
        r_grad = np.zeros_like(r.data)
        shape = np.broadcast_shapes(l.data.shape, r.data.shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                l_data[wrap(l_data,i,j)] -= self.delta
                v0 = l_data[wrap(l_data,i,j)] - r.data[wrap(r_data,i,j)]
                l_data[wrap(l_data,i,j)] = l.data[wrap(l_data,i,j)]
                l_data[wrap(l_data,i,j)] += self.delta
                v1 = l_data[wrap(l_data,i,j)] - r.data[wrap(r_data,i,j)]
                l_data[wrap(l_data,i,j)] = l.data[wrap(l_data,i,j)]
                l_grad[wrap(l_data,i,j)] += (v1 - v0) / (2.0 * self.delta)
                r_data[wrap(r_data,i,j)] -= self.delta
                v0 = l.data[wrap(l_data,i,j)] - r_data[wrap(r_data,i,j)]
                r_data[wrap(r_data,i,j)] = r.data[wrap(r_data,i,j)]
                r_data[wrap(r_data,i,j)] += self.delta
                v1 = l.data[wrap(l_data,i,j)] - r_data[wrap(r_data,i,j)]
                r_data[wrap(r_data,i,j)] = r.data[wrap(r_data,i,j)]
                r_grad[wrap(r_data,i,j)] += (v1 - v0) / (2.0 * self.delta)
        return "\n".join((
                self.log_entry("sub.l", 
                    np.sqrt(np.sum(np.power(l.grad - l_grad, 2.0))) /
                    np.sqrt(np.sum(np.power(l.grad + l_grad, 2.0)))),
                self.log_entry("sub.r", 
                    np.sqrt(np.sum(np.power(r.grad - r_grad, 2.0))) /
                    np.sqrt(np.sum(np.power(r.grad + r_grad, 2.0))))
                ))

    def mul(self) -> str:
        l = Tensor(np.random.rand(24, 1).astype(np.float64))
        r = Tensor(np.random.rand(1, 46).astype(np.float64))
        (l * r).backprop()
        l_data = l.data.copy()
        r_data = r.data.copy()
        l_grad = np.zeros_like(l.data)
        r_grad = np.zeros_like(r.data)
        shape = np.broadcast_shapes(l.data.shape, r.data.shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                l_data[wrap(l_data,i,j)] -= self.delta
                v0 = l_data[wrap(l_data,i,j)] * r.data[wrap(r_data,i,j)]
                l_data[wrap(l_data,i,j)] = l.data[wrap(l_data,i,j)]
                l_data[wrap(l_data,i,j)] += self.delta
                v1 = l_data[wrap(l_data,i,j)] * r.data[wrap(r_data,i,j)]
                l_data[wrap(l_data,i,j)] = l.data[wrap(l_data,i,j)]
                l_grad[wrap(l_data,i,j)] += (v1 - v0) / (2.0 * self.delta)
                r_data[wrap(r_data,i,j)] -= self.delta
                v0 = l.data[wrap(l_data,i,j)] * r_data[wrap(r_data,i,j)]
                r_data[wrap(r_data,i,j)] = r.data[wrap(r_data,i,j)]
                r_data[wrap(r_data,i,j)] += self.delta
                v1 = l.data[wrap(l_data,i,j)] * r_data[wrap(r_data,i,j)]
                r_data[wrap(r_data,i,j)] = r.data[wrap(r_data,i,j)]
                r_grad[wrap(r_data,i,j)] += (v1 - v0) / (2.0 * self.delta)
        return "\n".join((
                self.log_entry("mul.l", 
                    np.sqrt(np.sum(np.power(l.grad - l_grad, 2.0))) /
                    np.sqrt(np.sum(np.power(l.grad + l_grad, 2.0)))),
                self.log_entry("mul.r", 
                    np.sqrt(np.sum(np.power(r.grad - r_grad, 2.0))) /
                    np.sqrt(np.sum(np.power(r.grad + r_grad, 2.0))))
                ))

    def div(self) -> str:
        l = Tensor(np.random.rand(24, 1).astype(np.float64))
        r = Tensor(np.random.rand(1, 46).astype(np.float64))
        (l / r).backprop()
        l_data = l.data.copy()
        r_data = r.data.copy()
        l_grad = np.zeros_like(l.data)
        r_grad = np.zeros_like(r.data)
        shape = np.broadcast_shapes(l.data.shape, r.data.shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                l_data[wrap(l_data,i,j)] -= self.delta
                v0 = l_data[wrap(l_data,i,j)] / r.data[wrap(r_data,i,j)]
                l_data[wrap(l_data,i,j)] = l.data[wrap(l_data,i,j)]
                l_data[wrap(l_data,i,j)] += self.delta
                v1 = l_data[wrap(l_data,i,j)] / r.data[wrap(r_data,i,j)]
                l_data[wrap(l_data,i,j)] = l.data[wrap(l_data,i,j)]
                l_grad[wrap(l_data,i,j)] += (v1 - v0) / (2.0 * self.delta)
                r_data[wrap(r_data,i,j)] -= self.delta
                v0 = l.data[wrap(l_data,i,j)] / r_data[wrap(r_data,i,j)]
                r_data[wrap(r_data,i,j)] = r.data[wrap(r_data,i,j)]
                r_data[wrap(r_data,i,j)] += self.delta
                v1 = l.data[wrap(l_data,i,j)] / r_data[wrap(r_data,i,j)]
                r_data[wrap(r_data,i,j)] = r.data[wrap(r_data,i,j)]
                r_grad[wrap(r_data,i,j)] += (v1 - v0) / (2.0 * self.delta)
        return "\n".join((
                self.log_entry("div.l", 
                    np.sqrt(np.sum(np.power(l.grad - l_grad, 2.0))) /
                    np.sqrt(np.sum(np.power(l.grad + l_grad, 2.0)))),
                self.log_entry("div.r", 
                    np.sqrt(np.sum(np.power(r.grad - r_grad, 2.0))) /
                    np.sqrt(np.sum(np.power(r.grad + r_grad, 2.0))))
                ))
    
    def pow(self) -> str:
        l = Tensor(np.random.rand(24, 1).astype(np.float64))
        r = Tensor(np.random.rand(1, 46).astype(np.float64))
        (l ** r).backprop()
        l_data = l.data.copy()
        r_data = r.data.copy()
        l_grad = np.zeros_like(l.data)
        r_grad = np.zeros_like(r.data)
        shape = np.broadcast_shapes(l.data.shape, r.data.shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                l_data[wrap(l_data,i,j)] -= self.delta
                v0 = l_data[wrap(l_data,i,j)] ** r.data[wrap(r_data,i,j)]
                l_data[wrap(l_data,i,j)] = l.data[wrap(l_data,i,j)]
                l_data[wrap(l_data,i,j)] += self.delta
                v1 = l_data[wrap(l_data,i,j)] ** r.data[wrap(r_data,i,j)]
                l_data[wrap(l_data,i,j)] = l.data[wrap(l_data,i,j)]
                l_grad[wrap(l_data,i,j)] += (v1 - v0) / (2.0 * self.delta)
                r_data[wrap(r_data,i,j)] -= self.delta
                v0 = l.data[wrap(l_data,i,j)] ** r_data[wrap(r_data,i,j)]
                r_data[wrap(r_data,i,j)] = r.data[wrap(r_data,i,j)]
                r_data[wrap(r_data,i,j)] += self.delta
                v1 = l.data[wrap(l_data,i,j)] ** r_data[wrap(r_data,i,j)]
                r_data[wrap(r_data,i,j)] = r.data[wrap(r_data,i,j)]
                r_grad[wrap(r_data,i,j)] += (v1 - v0) / (2.0 * self.delta)
        return "\n".join((
                self.log_entry("pow.l", 
                    np.sqrt(np.sum(np.power(l.grad - l_grad, 2.0))) /
                    np.sqrt(np.sum(np.power(l.grad + l_grad, 2.0)))),
                self.log_entry("pow.r", 
                    np.sqrt(np.sum(np.power(r.grad - r_grad, 2.0))) /
                    np.sqrt(np.sum(np.power(r.grad + r_grad, 2.0))))
                ))

    def matmul(self) -> str:
        l = Tensor(np.random.rand(24, 36).astype(np.float64))
        r = Tensor(np.random.rand(36, 48).astype(np.float64))
        (l @ r).backprop()
        l_data = l.data.copy()
        r_data = r.data.copy()
        l_grad = np.zeros_like(l.data)
        r_grad = np.zeros_like(r.data)
        for i in range(l.data.shape[0]):
            for j in range(r.data.shape[1]):
                for k in range(l.data.shape[1]):
                    l_data[i,k] -= self.delta
                    v0 = np.dot(l_data[i,:], r.data[:,j])
                    l_data[i,k] = l.data[i,k]
                    l_data[i,k] += self.delta
                    v1 = np.dot(l_data[i,:], r.data[:,j])
                    l_data[i,k] = l.data[i,k]
                    l_grad[i,k] += (v1 - v0) / (2.0 * self.delta)
                for k in range(r.data.shape[0]):
                    r_data[k,j] -= self.delta
                    v0 = np.dot(l.data[i,:], r_data[:,j])
                    r_data[k,j] = r.data[k,j]
                    r_data[k,j] += self.delta
                    v1 = np.dot(l.data[i,:], r_data[:,j])
                    r_data[k,j] = r.data[k,j]
                    r_grad[k,j] += (v1 - v0) / (2.0 * self.delta)
        return "\n".join((
                self.log_entry("matmul.l", 
                    np.sqrt(np.sum(np.power(l.grad - l_grad, 2.0))) /
                    np.sqrt(np.sum(np.power(l.grad + l_grad, 2.0)))),
                self.log_entry("matmul.r", 
                    np.sqrt(np.sum(np.power(r.grad - r_grad, 2.0))) /
                    np.sqrt(np.sum(np.power(r.grad + r_grad, 2.0))))
                ))
    



if __name__ == "__main__":
    with GradCheck() as gc:
        print(gc.abs())
        print(gc.max())
        print(gc.neg())
        print(gc.square())
        print(gc.log())
        print(gc.exp())
        print(gc.sqrt())
        print(gc.tanh())
        print(gc.sum())
        print(gc.mean())
        print(gc.maximum())
        print(gc.add())
        print(gc.sub())
        print(gc.mul())
        print(gc.div())
        print(gc.pow())
        print(gc.matmul())
