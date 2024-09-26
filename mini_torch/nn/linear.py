import mini_torch
import mini_torch.nn as nn


class Linear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(mini_torch.randn(in_dim, out_dim))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(mini_torch.zeros((1, out_dim)))

    def forward(self, x):
        x = mini_torch.matmul(x, self.weight)
        if self.bias:
            x += self.bias
        return x
