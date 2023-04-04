import mini_torch.nn as nn
import mini_torch.nn.functional as F


class Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class CrossEntropyLoss(Loss):
    def __init__(self):
        super().__init__()

    def forward(self, x, target):
        x = x.reshape(-1, x.shape[-1])
        target = target.reshape(-1)
        return F.cross_entropy(x, target)
