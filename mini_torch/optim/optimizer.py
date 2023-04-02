class Optimizer:
    def __init__(self, parameters, learning_rate):
        self.parameters = list(parameters)
        self.lr = learning_rate

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for param in self.parameters:
            param.grad = 0.0