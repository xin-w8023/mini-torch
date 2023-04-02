import queue

import numpy as np

from mini_torch.tensor import Tensor


def backward(tensor: Tensor):

    def build_topo():
        topo = []
        q = queue.Queue()
        q.put(tensor)
        while not q.empty():
            t = q.get()
            if t in topo:
                topo.remove(t)
            topo.append(t)
            for e in t.pre:
                if e in topo:
                    topo.remove(e)
                q.put(e)
        return topo

    topo = build_topo()

    # zero grad
    for t in topo:
        t.grad = np.zeros_like(t.data)

    tensor.grad = np.ones_like(tensor.data)
    for t in topo:
        if t.requires_grad and t.grad_fn:
            t.grad_fn()
