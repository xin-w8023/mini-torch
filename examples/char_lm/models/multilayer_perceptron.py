from mini_torch import nn


class MultiLayerPerceptron(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x).reshape(x.shape[0], -1)
        return self.linear(x)
