from mini_torch.tensor import Tensor


def test_create_tensor():
    x = [1, 2, 3]
    x_tensor = Tensor(x)
    print(x_tensor)
