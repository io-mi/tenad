from typing import override
import gzip
import numpy as np
from numpy.typing import NDArray

from tenad.tensor import Tensor
from tenad.module import Module
from tenad.layer import Linear, Bias1D
from tenad.activation import ReLU, Softmax
from tenad.criterion import CrossEntropyLoss
from tenad.optimizer import StochasticGradientDescent


def load_idx(path:str) -> NDArray:
    with gzip.open(path) as idx:
        idx.seek(2) # skip zero bytes
        dtype = (np.uint8, np.int8, None, np.int16, np.int32, np.float32, np.float64)[int.from_bytes(idx.read(1), "big") - 8]
        ndims = int.from_bytes(idx.read(1), "big")
        shape = tuple(int.from_bytes(idx.read(4), "big") for _ in range(ndims))
        data = np.empty(shape, dtype)
        idx.readinto(data)
        return data


class Model(Module):
    def __init__(self) -> None:
        self.l0 = Linear(784, 128)
        self.b0 = Bias1D(128)
        self.a0 = ReLU()
        self.l1 = Linear(128, 10)
        self.b1 = Bias1D(10)
        self.a1 = Softmax()

    @override
    def __call__(self, x: Tensor) -> Tensor:
        for module in self.modules():
            x = module(x)
        return x


if __name__ == "__main__":
    np.seterr(invalid="raise")

    train_x = load_idx("demo/data/mnist/train-images-idx3-ubyte.gz")
    train_y = load_idx("demo/data/mnist/train-labels-idx1-ubyte.gz")

    sample_count = train_y.shape[0]
    epoch_count = 20
    batch_size = 50
    batch_count = sample_count / batch_size

    xs = train_x.astype(np.float32) / 255
    ys = np.zeros((sample_count,10), dtype=np.float32)
    ys[np.arange(sample_count), train_y] = 1

    model = Model()
    loss = CrossEntropyLoss()

    optimizer = StochasticGradientDescent(model.parameters(), 0.1)

    indices = np.arange(sample_count)
    for epoch in range(epoch_count):
        np.random.shuffle(indices)

        running_loss = Tensor(np.zeros(1, Tensor.dtype), False)
        for i in range(0, sample_count, batch_size):
            batch_indices = indices[i:i+batch_size]
            x = Tensor(xs[batch_indices,:].reshape(-1, 784), False)
            y = Tensor(ys[batch_indices,:], False)

            p = model(x)
            l = loss(p, y)
            l.backprop()

            optimizer.step()
            model.zero_grad()

            running_loss += l

        running_loss /= batch_count
        print("loss    : {:6.4f}".format(running_loss.data[0]))
    

    test_x = load_idx("demo/data/mnist/t10k-images-idx3-ubyte.gz")
    test_y = load_idx("demo/data/mnist/t10k-labels-idx1-ubyte.gz")

    xs = Tensor(test_x.reshape((10000, 784)).astype(np.float32) / 255.0, False)
    ps = model(xs)

    pred_y = np.argmax(ps.data, axis=1, keepdims=True)
    accuracy = np.mean(pred_y == test_y.reshape((-1, 1)))
    print("accuracy: {:6.4f}".format(accuracy))
