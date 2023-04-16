from .optimizer import Optimizer


class LRScheduler:
    def __init__(
        self,
        optimizer: Optimizer,
    ):
        self.optimizer = optimizer
        self._step = -1
        self._epoch = 0

    def step(self):
        raise NotImplementedError

    def get_last_lr(self):
        raise NotImplementedError


class StepScheduler(LRScheduler):
    def __init__(
        self,
        optimizer,
        lr=1e-3,
        factor=0.99,
        interval_step=1000,
        interval_type="step",
    ):
        super().__init__(optimizer)
        self.lr = lr
        self.interval_step = interval_step
        self.interval_type = interval_type
        self.factor = factor

    def step(self, epoch=0):
        if self.interval_type == "epoch":
            if epoch != self._epoch and epoch % self.interval_step == 0:
                self._epoch = epoch
                self.lr *= self.factor
                self.optimizer.lr = self.lr
        else:
            self._step += 1
            if self._step % self.interval_step == 0:
                self.lr *= self.factor
                self.optimizer.lr = self.lr

    def get_last_lr(self):
        return self.lr
