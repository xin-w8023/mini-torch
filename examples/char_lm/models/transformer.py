from mini_torch import nn


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_size,
        num_heads,
        hidden_size,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attn = nn.SelfAttention(embed_size, num_heads)
