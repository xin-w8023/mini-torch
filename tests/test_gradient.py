import pytest
import torch

import mini_torch


def test_embedding_gradient():
    torch.manual_seed(42)

    torch_embed = torch.nn.Embedding(4, 3)
    torch_inputs = torch.randint(0, 4, (2, 3))

    embed = mini_torch.nn.Embedding(4, 3)
    embed.weight.data = torch_embed.weight.detach().numpy()
    inputs = mini_torch.Tensor(torch_inputs.detach().numpy(), dtype=int)

    # torch forward
    torch_x = torch_embed(torch_inputs)
    torch_loss = torch_x.mean()
    torch_loss.backward()

    # mini-torch forward
    x = embed(inputs)
    loss = x.mean()
    loss.backward()

    # comparison
    torch.testing.assert_allclose(torch_loss.item(), loss.item())
    torch.testing.assert_allclose(torch_embed.weight.grad, embed.weight.grad)


def test_linear_gradient():
    torch.manual_seed(42)

    torch_linear = torch.nn.Linear(20, 10)
    torch_inputs = torch.randn(10, 20)

    linear = mini_torch.nn.Linear(20, 10)
    linear.weight.data = torch_linear.weight.detach().numpy().T
    linear.bias.data = torch_linear.bias.detach().numpy().reshape((1, -1))
    inputs = mini_torch.Tensor(torch_inputs.detach().numpy(), dtype=int)

    # torch forward
    torch_x = torch_linear(torch_inputs)
    torch_loss = torch_x.mean()
    torch_loss.backward()

    # mini-torch forward
    x = linear(inputs)
    loss = x.mean()
    loss.backward()

    # comparison
    torch.testing.assert_allclose(torch_loss.item(), loss.item())
    torch.testing.assert_allclose(torch_linear.weight.grad.T, linear.weight.grad)
    torch.testing.assert_allclose(torch_linear.bias.grad.view(1, -1), linear.bias.grad)


def test_ce_gradient():
    torch.manual_seed(42)

    bs, dim = 20, 40

    torch_ce = torch.nn.CrossEntropyLoss()
    torch_x = torch.randn(bs, dim, requires_grad=True)
    torch_y = torch.randint(0, dim, (bs,))

    ce = mini_torch.nn.CrossEntropyLoss()
    x = mini_torch.Tensor(torch_x.detach().numpy(), requires_grad=True)
    y = mini_torch.Tensor(torch_y.detach().numpy(), dtype=int)

    # torch forward
    torch_loss = torch_ce(torch_x, torch_y)
    torch_loss.backward()

    # mini-torch forward
    loss = ce(x, y)
    loss.backward()

    # comparison
    torch.testing.assert_allclose(torch_loss.item(), loss.item())
    torch.testing.assert_allclose(torch_x.grad, x.grad)


def test_softmax_gradient():
    torch.manual_seed(42)

    shape = (2, 3, 4, 5)

    for dim in range(len(shape)):
        torch_x = torch.randn(*shape, requires_grad=True)
        x = mini_torch.Tensor(torch_x.detach().numpy(), requires_grad=True)

        # torch forward
        torch_loss = torch.nn.functional.softmax(torch_x, dim=dim).mean()
        torch_loss.backward()

        # mini-torch forward
        loss = mini_torch.nn.functional.softmax(x, dim=dim).mean()
        loss.backward()

        # comparison
        torch.testing.assert_allclose(torch_loss.item(), loss.item())
        torch.testing.assert_allclose(torch_x.grad, x.grad)


@pytest.mark.parametrize("B", [2, 4, 8, 16, 32])
@pytest.mark.parametrize("T", [128, 256])
@pytest.mark.parametrize("D", [32, 64])
def test_batch_matmul_gradient(B, T, D):
    torch.manual_seed(42)

    torch_x1 = torch.randn(B, T, D, requires_grad=True)
    torch_x2 = torch.randn(B, D, T, requires_grad=True)
    x1 = mini_torch.Tensor(torch_x1.detach().numpy(), requires_grad=True)
    x2 = mini_torch.Tensor(torch_x2.detach().numpy(), requires_grad=True)

    # torch forward
    torch_loss = torch.bmm(torch_x1, torch_x2).mean()
    torch_loss.backward()

    # mini-torch forward
    loss = mini_torch.bmm(x1, x2).mean()
    loss.backward()

    # comparison
    torch.testing.assert_allclose(torch_loss.item(), loss.item())
    torch.testing.assert_allclose(torch_x1.grad, x1.grad)
    torch.testing.assert_allclose(torch_x2.grad, x2.grad)
