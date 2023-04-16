import numpy as np
from torch.utils.tensorboard import SummaryWriter

import mini_torch.nn as nn
import mini_torch.optim
from mini_torch.meter.record import Record


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


example_dir = "examples/char_lm"
data = open(f"{example_dir}/t8.shakespeare.txt", "r", encoding="utf8").read()
chars = sorted(set(data))

c2i = {c: i for i, c in enumerate(chars)}
c2i["<S>"] = len(c2i)
i2c = {i: c for c, i in c2i.items()}

data = [c2i[e] for e in data]

vocab_size = len(c2i)
embed_size = vocab_size
prompt = " " * 16
block_size = len(prompt)
batch_size = 32

model = Model(vocab_size, embed_size, block_size * embed_size)
criterion = nn.CrossEntropyLoss()
opt = mini_torch.optim.SGD(model.parameters(), 0.5)
scheduler = mini_torch.optim.StepScheduler(
    opt, lr=0.5, factor=0.5, interval_step=1, interval_type="epoch"
)
recoder = Record()

writer = SummaryWriter(f"{example_dir}/log/epoch")

global_step = 0
for epoch in range(10000):
    recoder.reset()
    for x, t in batcher(data, block_size, c2i["<S>"], batch_size=batch_size):
        y = model(x)
        loss = criterion(y, t)

        opt.zero_grad()
        loss.backward()
        opt.step()
        recoder.append(loss.item())
        scheduler.step(epoch=epoch)
        global_step += 1
        if global_step % 1000 == 0:
            writer.add_scalar("loss", recoder.report(), global_step)
            writer.add_scalar("lr", scheduler.get_last_lr(), global_step)
            print(global_step / 1000, recoder.report())

            text = ""
            xl = [int(c2i[e]) for e in prompt]
            for _ in range(1024):
                p = mini_torch.Tensor(xl, dtype=int).reshape(1, -1)
                p = model(p).data.reshape(-1)
                p = p - np.max(p)
                p = np.exp(p) / np.sum(np.exp(p))
                p = np.random.choice(list(range(vocab_size)), size=1, p=p)[0]
                text += i2c[p]
                xl = xl[1:] + [p]
            writer.add_text("text", text, global_step)
