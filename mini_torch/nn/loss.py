import mini_torch.nn as nn
import mini_torch.nn.functinoal as F


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
        # xmax = x.max(1, keepdims=True)
        # xmax = -xmax
        # x = x + xmax
        # exp = x.exp()
        # exp_sum = exp.sum(1, keepdims=True)
        # exp_sum = exp_sum ** -1
        # x = exp * exp_sum
        # log_softmax = torch.log(x)
        # loss = log_softmax[:, target]
        # loss = -loss
        # if self.reduction == "sum":
        #     loss = loss.sum()
        # elif self.reduction == "mean":
        #     loss = loss.mean()
        # return loss
