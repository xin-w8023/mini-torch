import numpy as np

import mini_torch


class TextDataset:
    def __init__(self, filename, block_size, batch_size):
        self.filename = filename
        self.block_size = block_size
        self.batch_size = batch_size
        self.raw_data = open(self.filename, "r", encoding="utf8").read()

        self._chars = sorted(set(self.raw_data))

        self._c2i = {c: i for i, c in enumerate(self._chars)}
        self._i2c = {i: c for c, i in self._c2i.items()}

    @property
    def vocab_size(self):
        return len(self._c2i)

    def encode(self, texts):
        return mini_torch.Tensor([self._c2i[c] for c in texts], dtype=int)

    def decode(self, codes):
        return "".join([self._i2c[i] for i in codes])

    def draw_start(self, size):
        s = np.random.randint(0, len(self.raw_data) - size)
        return self.raw_data[s : s + size]

    def __iter__(self):
        x = list(self.raw_data[: self.block_size])
        mini_batch = []
        for v in self.raw_data[self.block_size :]:
            mini_batch.append(
                (
                    self.encode(x).reshape((1, -1)),
                    self.encode([v]),
                )
            )
            if len(mini_batch) == self.batch_size:
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
