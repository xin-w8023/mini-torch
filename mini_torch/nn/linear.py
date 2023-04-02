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


if __name__ == '__main__':
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.li = Linear(10, 20)

        def forward(self, x):
            return self.li(x)

    m = Model()
    print(dict(m.named_parameters()))
    print(m)
