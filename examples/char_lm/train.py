import numpy as np
from torch.utils.tensorboard import SummaryWriter

import mini_torch.nn as nn
import mini_torch.optim


class Model(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x).reshape(x.shape[0], -1)
        return self.linear(x)


def batcher(data, block_size=5, bos=0, batch_size=1):
    x = [bos] * block_size
    mini_batch = []
    for i, v in enumerate(data):
        mini_batch.append(
            (
                mini_torch.Tensor(x, dtype=int).reshape((1, -1)),
                mini_torch.Tensor(v, dtype=int),
            )
        )
        if len(mini_batch) == batch_size:
            xs = mini_torch.cat([i for i, v in mini_batch], dim=0)
            ys = mini_torch.cat([v for i, v in mini_batch])
            yield xs, ys
            mini_batch = []
        x.append(v)
        x = x[1:]
    if mini_batch:
        xs = mini_torch.cat([i for i, v in mini_batch], dim=0)
        ys = mini_torch.cat([v for i, v in mini_batch])
        yield xs, ys


data = open("code.txt", "r", encoding="utf8").read()
chars = sorted(set(data))

c2i = {c: i for i, c in enumerate(chars)}
c2i["<S>"] = len(c2i)
i2c = {i: c for c, i in c2i.items()}

data = [c2i[e] for e in data]

vocab_size = len(c2i)
embed_size = vocab_size
prompt = "            "
block_size = len(prompt)
batch_size = 20

model = Model(vocab_size, embed_size, block_size * embed_size)
criterion = nn.CrossEntropyLoss()
opt = mini_torch.optim.SGD(model.parameters(), 0.1)


writer = SummaryWriter("./log")

global_step = 0
for _ in range(10000):
    for x, t in batcher(data, block_size, c2i["<S>"], batch_size=batch_size):
        y = model(x)
        loss = criterion(y, t)

        opt.zero_grad()
        loss.backward()
        opt.step()
        if global_step % 1000:
            writer.add_scalar("loss", loss.item(), global_step)
        global_step += 1
    print(loss.item())
    text = ""
    xl = [int(c2i[e]) for e in prompt]
    for _ in range(100):
        x = mini_torch.Tensor(xl, dtype=int).reshape(1, -1)
        p = model(x).data.reshape(-1)
        p = p - np.max(p)
        p = np.exp(p) / np.sum(np.exp(p))
        x = np.random.choice(list(range(vocab_size)), size=1, p=p)[0]
        text += i2c[x]
        xl = xl[1:] + [x]
    writer.add_text("text", text, global_step)
