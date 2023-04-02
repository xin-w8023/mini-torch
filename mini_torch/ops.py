import mini_torch
from mini_torch.backward_functions import *

import numpy as np


def exp(x):
    out = x.clone()
    out.data = np.exp(out.data)
    out.grad_fn = ExpBackwardFunction(x, out)
    return out


def log(x):
    out = x.clone()
    out.data = np.log(out.data)

    out.grad_fn = LogBackwardFunction(x, out)
    return out


def max(x, dim=None, keepdims=False):
    if dim is None:
        out = mini_torch.Tensor(np.max(x.data, keepdims=keepdims), x.dtype, x.requires_grad)
    else:
        out = mini_torch.Tensor(np.max(x.data, axis=dim, keepdims=keepdims), x.dtype, x.requires_grad)

    out.grad_fn = MaxBackwardFunction(x, out, dim)
    return out


def sum(x, dim=None, keepdims=False):
    if dim is None:
        out = mini_torch.Tensor(np.sum(x.data, keepdims=keepdims), x.dtype, x.requires_grad)
    else:
        out = mini_torch.Tensor(np.sum(x.data, axis=dim, keepdims=keepdims), x.dtype, x.requires_grad)
    out.grad_fn = SumBackwardFunction(x, out)
    return out


def matmul(x, other):
    requires_grad = x.requires_grad or other.requires_grad
    out = mini_torch.Tensor(x.data.dot(other.data), requires_grad=requires_grad)
    out.grad_fn = MatMulBackwardFunction(x, other, out)
    return out


def reshape(x, *dims):
    out = x.clone()
    out.data = out.data.reshape(*dims)
    out.grad_fn = ReshapeBackwardFunction(x, out)
    return out


def pow(x, power):
    out = x.clone()
    out.data = out.data ** power
    out.grad_fn = PowBackwardFunction(x, out, power)
    return out


def index_select(x, index):
    out = x.clone()
    if isinstance(index, mini_torch.Tensor):
        slice = index.data
    elif isinstance(index, int):
        slice = index
    else:
        slice = tuple([i if not isinstance(i, mini_torch.Tensor) else i.data for i in index])
    out.data = out.data[slice]
    out.grad_fn = IndexSelectBackwardFunction(x, out, slice)
    return out


def mean(x, dims=None, keepdims=False):
    kwargs = {}
    if dims is not None:
        kwargs["axis"] = dims
    if keepdims is not None:
        kwargs["keepdims"] = keepdims
    data = x.data.mean(**kwargs)
    out = mini_torch.Tensor(data, requires_grad=x.requires_grad)
    out.grad_fn = MeanBackwardFunction(x, out, dims, keepdims)

    return out


def mul(x, other):
    requires_grad = x.requires_grad or other.requires_grad
    out = mini_torch.Tensor(x.data * other.data, requires_grad=requires_grad)
    out.grad_fn = MulBackwardFunction(x, other, out)
    return out


def add(x, other):
    requires_grad = x.requires_grad or other.requires_grad
    out = mini_torch.Tensor(x.data + other.data, requires_grad=requires_grad)
    out.grad_fn = AddBackwardFunction(x, other, out)
    return out


def neg(x):
    out = x.clone()
    out.data = -out.data
    out.grad_fn = NegBackwardFunction(x, out)
    return out


def cat(tensors, dim=None):
    datas = [t.data for t in tensors]
    datas = np.concatenate(datas, axis=dim)
    out = mini_torch.Tensor(datas)
    return out
