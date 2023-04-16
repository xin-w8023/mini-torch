import numpy as np
from torch.utils.tensorboard import SummaryWriter

import mini_torch.nn as nn
import mini_torch.optim
from mini_torch.meter.record import Record

from .data import TextDataset
from .models import MultiLayerPerceptron

example_dir = "examples/char_lm"
data_file = f"{example_dir}/t8.shakespeare.txt"

block_size = 16
batch_size = 32

dataset = TextDataset(data_file, block_size, batch_size)
vocab_size = dataset.vocab_size
embed_size = vocab_size
model = MultiLayerPerceptron(vocab_size, embed_size, block_size * embed_size)
criterion = nn.CrossEntropyLoss()
opt = mini_torch.optim.SGD(model.parameters(), 0.5)
scheduler = mini_torch.optim.StepScheduler(
    opt, lr=0.5, factor=0.5, interval_step=1, interval_type="epoch"
)
recoder = Record()

writer = SummaryWriter(f"{example_dir}/log/multilayer_perceptron")

global_step = 0
for epoch in range(10000):
    recoder.reset()
    for x, t in dataset:
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

            xl = dataset.encode(dataset.draw_start(block_size)).to_list()
            codes = xl[:]
            for _ in range(1024):
                p = mini_torch.Tensor(xl, dtype=int).reshape(1, -1)
                p = model(p).data.reshape(-1)
                p = p - np.max(p)
                p = np.exp(p) / np.sum(np.exp(p))
                p = np.random.choice(list(range(vocab_size)), size=1, p=p)[0]
                codes.append(p)
                xl = xl[1:] + [p]
            text = dataset.decode(codes)
            writer.add_text("text", text, global_step)
