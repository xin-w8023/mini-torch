import copy

import numpy as np

import mini_torch
import mini_torch.backward_functions as BF


def cross_entropy(x, t, reduction="mean"):
    data_max = x.data.max(1, keepdims=True)
    exp = np.exp(x.data - data_max)
    softmax = exp / exp.sum(axis=1, keepdims=True)
    loss = -np.log(softmax[range(x.shape[0]), t.data])
    if reduction == "sum":
        loss = loss.sum()
    else:
        loss = loss.mean()
    out = mini_torch.Tensor(loss, requires_grad=x.requires_grad)

    out.grad_fn = BF.NllLossBackwardFunction(x, out, softmax, t, reduction)

    return out


def relu(x):
    return leaky_relu(x, 0)


def leaky_relu(x, leaky=0.1):
    data = copy.deepcopy(x.data)
    data[x.data < 0] = leaky * data[x.data < 0]
    out = mini_torch.Tensor(data, requires_grad=x.requires_grad)
    out.grad_fn = BF.LeakyReluBackwardFunction(x, out, leaky)
    return out


def softmax(x, dim=-1):
    data_max = x.data.max(dim, keepdims=True)
    exp = np.exp(x.data - data_max)
    softmax = exp / exp.sum(axis=dim, keepdims=True)
    softmax = mini_torch.Tensor(softmax, requires_grad=x.requires_grad)
    softmax.grad_fn = BF.SoftmaxBackwardFunction(x, softmax, dim)
    return softmax
