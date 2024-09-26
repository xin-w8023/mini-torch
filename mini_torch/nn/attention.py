import mini_torch
import mini_torch.nn.functional as F
from mini_torch import Tensor, nn


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, need_weights=False, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.need_weights = need_weights
        self.scale = mini_torch.tensor(self.head_dim**-0.5)

        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        query: Tensor,
    ):
        qkv = self.in_proj(query)
        q, k, v = self._split_qkv(qkv)
        attn = mini_torch.bmm(q, k.transpose((0, 2, 1))) * self.scale

        attn_w = F.softmax(attn)
        output = mini_torch.bmm(attn, v)
        return self._merge_attn(output, attn_w)

    def _split_qkv(self, qkv: Tensor):
        b, t, d = qkv.shape
        qkv = (
            qkv.reshape(b, t, 3, self.num_heads, self.head_dim)
            .transpose((2, 0, 3, 1, 4))
            .reshape(3, b * self.num_heads, t, self.head_dim)
        )
        return qkv[0], qkv[1], qkv[2]

    def _merge_attn(self, attn, attn_w):
        t = attn.shape[1]
        attn = (
            attn.reshape(-1, self.num_heads, t, self.head_dim)
            .transpose((0, 2, 1, 3))
            .reshape(-1, t, self.embed_dim)
        )
        if self.need_weights:
            attn_w = attn_w.reshape(-1, self.num_heads, t, 1).transpose((0, 2, 1, 3))
        return attn, attn_w if self.need_weights else None


if __name__ == "__main__":
    attn = SelfAttention(16, 2, need_weights=True)
    x = mini_torch.ones((1, 1, 16))
    out = attn(x)
    print(out[0].shape, out[1].shape)
