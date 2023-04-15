import inspect

import numpy as np


class BackwardFunction:
    def save_variables(self):
        params = inspect.signature(self.__init__).parameters
        last_locals = inspect.currentframe().f_back.f_locals
        for attr, param in params.items():
            setattr(self, attr, last_locals[attr])

    def dump_variables(self):
        params = inspect.signature(self.__init__).parameters
        return (getattr(self, attr) for attr in params)


class ExpBackwardFunction(BackwardFunction):
    def __init__(self, x, out):
        x._append(out)
        self.save_variables()

    def __call__(self):
        x, out = self.dump_variables()
        x.grad += out.grad * out.data


class LogBackwardFunction(BackwardFunction):
    def __init__(self, x, out):
        x._append(out)
        self.save_variables()

    def __call__(self):
        x, out = self.dump_variables()
        x.grad += out.grad * (1 / x.data)


class MaxBackwardFunction(BackwardFunction):
    def __init__(self, x, out, dim):
        x._append(out)
        self.save_variables()

    def __call__(self):
        x, out, dim = self.dump_variables()
        local_grad = np.zeros_like(x.data)
        if dim is None:
            idx = np.argmax(x.data)
            local_grad = local_grad.reshape(-1)
            local_grad[idx] = 1
            local_grad = local_grad.reshape(x.data.shape)
        else:
            idx = np.argmax(x.data, axis=dim, keepdims=True)
            local_grad[:, idx] = 1
        x.grad += out.grad * local_grad


class SumBackwardFunction(BackwardFunction):
    def __init__(self, x, out):
        x._append(out)
        self.save_variables()

    def __call__(self):
        x, out = self.dump_variables()
        x.grad += out.grad * np.ones_like(x.data)


class MatMulBackwardFunction(BackwardFunction):
    def __init__(self, x, other, out):
        other._append(out)
        x._append(out)
        self.save_variables()

    def __call__(self):
        x, other, out = self.dump_variables()
        x.grad += out.grad.dot(other.data.T)
        other.grad += x.data.T.dot(out.grad)


class ReshapeBackwardFunction(BackwardFunction):
    def __init__(self, x, out):
        x._append(out)
        self.save_variables()

    def __call__(self):
        x, out = self.dump_variables()
        x.grad += out.grad.reshape(x.shape)


class PowBackwardFunction(BackwardFunction):
    def __init__(self, x, out, power):
        x._append(out)
        self.save_variables()

    def __call__(self):
        x, out, power = self.dump_variables()
        x.grad += out.grad * (power * x.data ** (power - 1))


class IndexSelectBackwardFunction(BackwardFunction):
    def __init__(self, x, out, slice):
        x._append(out)
        self.save_variables()

    def __call__(self):
        x, out, slice = self.dump_variables()
        for i, out_grad in zip(
            slice.reshape(-1), out.grad.reshape((-1, out.grad.shape[-1]))
        ):
            x.grad[i] += out_grad


class MeanBackwardFunction(BackwardFunction):
    def __init__(self, x, out, dims, keepdims):
        x._append(out)
        self.save_variables()

    def __call__(self):
        x, out, dims, keepdims = self.dump_variables()
        if dims is None:
            scale = x.data.size
        elif isinstance(dims, list):
            scale = 1
            for dim in dims:
                scale *= x.data.shape[dim]
        else:
            scale = x.data.shape[dims]

        x.grad += out.grad * (1 / scale)


class NegBackwardFunction(BackwardFunction):
    def __init__(self, x, out):
        x._append(out)
        self.save_variables()

    def __call__(self):
        x, out = self.dump_variables()
        x.grad += -out.grad


class AddBackwardFunction(BackwardFunction):
    def __init__(self, x, other, out):
        other._append(out)
        x._append(out)
        self.save_variables()
        self.reduce_dims = []
        for i, (ix1, ix2) in enumerate(zip(self.x.shape, self.other.shape)):
            if ix1 != ix2:
                self.reduce_dims.append(i)

    def __call__(self):
        x, other, out = self.dump_variables()
        grad = np.ones_like(x.data) * out.grad
        if grad.shape != x.data.shape:
            for dim in self.reduce_dims:
                grad = grad.sum(axis=dim, keepdims=True)
        x.grad += grad
        grad = np.ones_like(other.data) * out.grad
        if grad.shape != other.data.shape:
            for dim in self.reduce_dims:
                grad = grad.sum(axis=dim, keepdims=True)
        other.grad += grad


class MulBackwardFunction(BackwardFunction):
    def __init__(self, x, other, out):
        other._append(out)
        x._append(out)
        self.save_variables()
        self.reduce_dims = []
        for i, (ix1, ix2) in enumerate(zip(self.x.shape, self.other.shape)):
            if ix1 != ix2:
                self.reduce_dims.append(i)

    def __call__(self):
        x, other, out = self.dump_variables()
        grad = np.ones_like(x.data) * out.grad * other.data
        if grad.shape != x.data.shape:
            for dim in self.reduce_dims:
                grad = grad.sum(axis=dim, keepdims=True)
        x.grad += grad
        grad = np.ones_like(other.data) * out.grad * x.data
        if grad.shape != other.data.shape:
            for dim in self.reduce_dims:
                grad = grad.sum(axis=dim, keepdims=True)
        other.grad += grad


class NllLossBackwardFunction(BackwardFunction):
    def __init__(self, x, out, softmax, target, reduction="mean"):
        x._append(out)
        self.save_variables()

    def __call__(self):
        x, out, softmax, target, reduction = self.dump_variables()
        softmax[range(x.shape[0]), target.data] -= 1
        scale = 1
        if reduction == "mean":
            scale /= x.shape[0]
        x.grad += out.grad * softmax * scale


class LeakyReluBackwardFunction(BackwardFunction):
    def __init__(self, x, out, leaky):
        x._append(out)
        self.save_variables()

    def __call__(self):
        x, out, leaky = self.dump_variables()
        grad = np.ones_like(x.data)
        grad[x.data < 0] = -leaky
        x.grad += out.grad * grad


class SoftmaxBackwardFunction(BackwardFunction):
    def __init__(self, x, out, dim):
        x._append(out)
        self.save_variables()

    def __call__(self):
        x, out, dim = self.dump_variables()
        trans = np.arange(x.ndim)
        trans[dim] = -1
        trans[-1] = dim

        if x.ndim > 1:
            tmp = out.data.transpose(trans)
        else:
            tmp = out.data

        grad = np.zeros_like(tmp)

        for i in range(x.shape[dim]):
            grad[..., i] = tmp[..., i] - np.sum(tmp[..., i : i + 1] * tmp, axis=-1)

        if x.ndim > 1:
            grad = grad.transpose(trans)

        x.grad += out.grad * grad
