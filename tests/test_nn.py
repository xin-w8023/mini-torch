import numpy as np

import mini_torch
import mini_torch.graph
import mini_torch.nn as nn
import mini_torch.nn.functional as F


def test_nn():
    np.random.seed(42)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = mini_torch.nn.Linear(2, 3)
            self.l2 = mini_torch.nn.Linear(3, 10)

        def forward(self, x):
            x = self.l1(x)
            x = F.relu(x)
            x = self.l2(x)
            return x

    m = Model()
    sgd = mini_torch.optim.SGD(m.parameters(), 0.1, momentum=0.8)
    criterion = mini_torch.nn.CrossEntropyLoss()

    x = mini_torch.randn(10, 2)
    t = mini_torch.Tensor([[5]] * 10, dtype=int)
    for _ in range(10):
        y = m(x)
        loss = criterion(y, t)
        print(loss.item())

        sgd.zero_grad()
        loss.backward()
        sgd.step()


if __name__ == "__main__":
    test_nn()
