import numpy as np

from mini_torch.tensor import Tensor


def tensor(data):
    return Tensor(data)


def randn(*dims, dtype=np.float32):
    data = np.random.randn(*dims)
    return Tensor(data, dtype=dtype)


def zeros(*dims, dtype=np.float32):
    return Tensor(np.zeros(*dims, dtype=dtype), dtype=dtype)


def ones(*dims, dtype=np.float32):
    return Tensor(np.ones(*dims, dtype=dtype), dtype=dtype)
