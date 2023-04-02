import mini_torch


class Module:
    def __init__(self, *args, **kwargs):
        pass

    def __repr__(self):
        return "repr: Need Update"

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def parameters(self, recurse=True):
        for name, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(self, prefix="", recurse=True):
        for item in self.__dict__:
            param = getattr(self, item)
            if isinstance(param, mini_torch.nn.Parameter):
                yield ".".join([prefix, item]), param
            elif isinstance(param, Module):
                if prefix:
                    prefix += "." + item
                else:
                    prefix = item
                for name, sub_param in param.named_parameters(prefix, recurse):
                    yield name, sub_param

    def _call_impl(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    __call__ = _call_impl
