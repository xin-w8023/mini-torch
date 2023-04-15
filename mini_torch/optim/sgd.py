import numpy as np

from mini_torch.optim.optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, parameters, lr=1e-3, momentum=0.9):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.v = [np.zeros_like(param.data) for param in self.parameters]

    def step(self):
        for v, param in zip(self.v, self.parameters):
            v = v * self.momentum + param.grad * (1 - self.momentum)
            param.data -= self.lr * v
