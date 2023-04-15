import mini_torch
from mini_torch import nn


class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.weight = nn.Parameter(mini_torch.randn(vocab_size, embed_size))

    def forward(self, x):
        return self.weight[x]
