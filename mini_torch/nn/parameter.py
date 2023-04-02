import mini_torch


class Parameter(mini_torch.Tensor):
    def __init__(self, data=None, requires_grad=True):
        if data is None:
            data = []
        if isinstance(data, mini_torch.Tensor):
            data = data.data
        super().__init__(data=data, requires_grad=requires_grad)
