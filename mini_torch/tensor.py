import copy
import mini_torch
import numpy as np


class Tensor:
    def __init__(self, data, dtype=np.float32, requires_grad=False, grad_fn=None):

        if not isinstance(data, np.ndarray):
            try:
                data = np.array(data, dtype=dtype)
            except Exception as e:
                raise ValueError(f"Cannot convert data with type {type(data)} to Tensor.")

        self.data = data
        self.dtype = dtype
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = grad_fn

        self.pre = []
        self.next = []

    def __repr__(self):
        return f"Tensor(" \
               f"{self.data}, " \
               f"dtype={self.dtype.__name__}, " \
               f"requires_grad={self.requires_grad}, " \
               f"grad_fn={self.grad_fn}" \
               f")"

    def _append(self, other):
        if other not in self.next:
            self.next.append(other)
        if self not in other.pre:
            other.pre.append(self)

    def __add__(self, other):
        return mini_torch.add(self, other)

    def __neg__(self):
        return mini_torch.neg(self)

    def __sub__(self, other):
        return self + -other

    def __iter__(self):
        for d in self.data:
            yield d

    def __mul__(self, other):
        return mini_torch.mul(self, other)

    def mean(self, dims=None, keepdims=False):
        return mini_torch.mean(self, dims, keepdims)

    def item(self):
        return self.data.item()

    def clone(self):
        return Tensor(copy.deepcopy(self.data), self.dtype,
                      self.requires_grad, grad_fn=self.grad_fn)

    def backward(self):
        mini_torch.autograd.backward(self)

    def reshape(self, *dims):
        return mini_torch.reshape(self, *dims)

    def max(self, dim=None, keepdims=False):
        return mini_torch.max(self, dim, keepdims)

    def exp(self):
        return mini_torch.exp(self)

    def sum(self, dim=None, keepdims=False):
        return mini_torch.sum(self, dim, keepdims)

    def __pow__(self, power, modulo=None):
        return mini_torch.pow(self, power)

    def __getitem__(self, index):
        return mini_torch.index_select(self, index)

    def __radd__(self, other):
        self.data += other
        return self

    def __rmul__(self, other):
        self.data *= other
        return self

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def shape(self):
        return self.data.shape


if __name__ == '__main__':
    import mini_torch.graph
    x = np.arange(1, 3).reshape(1, 2)
    y = np.arange(1, 3).reshape(2, 1)
    x = Tensor(x, label="x")
    y = Tensor(y, label="y")
    z = x * y
    z.backward()
    mini_torch.graph.show(z)
    # a = Tensor(3, label="a")
    # z = z * a
    # z.backward()
