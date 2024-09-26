import numpy as np

import mini_torch
import mini_torch.backward_functions as BF


def exp(x):
    out = x.clone()
    out.data = np.exp(out.data)
    out.grad_fn = BF.ExpBackwardFunction(x, out)
    return out


def log(x):
    out = x.clone()
    out.data = np.log(out.data)

    out.grad_fn = BF.LogBackwardFunction(x, out)
    return out


def max(x, dim=None, keepdims=False):
    if dim is None:
        out = mini_torch.Tensor(
            np.max(x.data, keepdims=keepdims), x.dtype, x.requires_grad
        )
    else:
        out = mini_torch.Tensor(
            np.max(x.data, axis=dim, keepdims=keepdims), x.dtype, x.requires_grad
        )

    out.grad_fn = BF.MaxBackwardFunction(x, out, dim)
    return out


def sum(x, dim=None, keepdims=False):
    if dim is None:
        out = mini_torch.Tensor(
            np.sum(x.data, keepdims=keepdims), x.dtype, x.requires_grad
        )
    else:
        out = mini_torch.Tensor(
            np.sum(x.data, axis=dim, keepdims=keepdims), x.dtype, x.requires_grad
        )
    out.grad_fn = BF.SumBackwardFunction(x, out)
    return out


def matmul(x, other):
    requires_grad = x.requires_grad or other.requires_grad
    out = mini_torch.Tensor(x.data.dot(other.data), requires_grad=requires_grad)
    out.grad_fn = BF.MatMulBackwardFunction(x, other, out)
    return out


def reshape(x, *dims):
    out = x.clone()
    out.data = out.data.reshape(*dims)
    out.grad_fn = BF.ReshapeBackwardFunction(x, out)
    return out


def pow(x, power):
    out = x.clone()
    out.data = out.data**power
    out.grad_fn = BF.PowBackwardFunction(x, out, power)
    return out


def index_select(x, index):
    out = x.clone()
    if isinstance(index, mini_torch.Tensor):
        s = index.data
    elif isinstance(index, int):
        s = index
    else:
        s = tuple(
            [i if not isinstance(i, mini_torch.Tensor) else i.data for i in index]
        )
    out.data = out.data[s]
    out.grad_fn = BF.IndexSelectBackwardFunction(x, out, s)
    return out


def mean(x, dims=None, keepdims=False):
    kwargs = {}
    if dims is not None:
        kwargs["axis"] = dims
    if keepdims is not None:
        kwargs["keepdims"] = keepdims
    data = x.data.mean(**kwargs)
    out = mini_torch.Tensor(data, requires_grad=x.requires_grad)
    out.grad_fn = BF.MeanBackwardFunction(x, out, dims, keepdims)

    return out


def mul(x, other):
    requires_grad = x.requires_grad or other.requires_grad
    out = mini_torch.Tensor(x.data * other.data, requires_grad=requires_grad)
    out.grad_fn = BF.MulBackwardFunction(x, other, out)
    return out


def add(x, other):
    requires_grad = x.requires_grad or other.requires_grad
    out = mini_torch.Tensor(x.data + other.data, requires_grad=requires_grad)
    out.grad_fn = BF.AddBackwardFunction(x, other, out)
    return out


def neg(x):
    out = x.clone()
    out.data = -out.data
    out.grad_fn = BF.NegBackwardFunction(x, out)
    return out


def cat(tensors, dim=None):
    datas = [t.data for t in tensors]
    datas = np.concatenate(datas, axis=dim)
    out = mini_torch.Tensor(datas)
    return out


def transpose(x, axis=None):
    out = x.clone()
    out.data = np.transpose(out.data, axis)
    out.grad_fn = BF.TransposeBackwardFunction(x, out, axis)
    return out


def bmm(x, other):
    requires_grad = x.requires_grad or other.requires_grad
    out = mini_torch.Tensor(np.matmul(x.data, other.data), requires_grad=requires_grad)
    out.grad_fn = BF.BatchMatMulBackwardFunction(x, other, out)
    return out
