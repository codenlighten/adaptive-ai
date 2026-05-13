import torch

from src.ternary import BitLinear, ste_ternarize, ternarize


def test_ternarize_values_are_in_set():
    torch.manual_seed(0)
    w = torch.randn(64, 64)
    w_q, alpha = ternarize(w)
    unique = torch.unique(w_q).tolist()
    assert set(unique).issubset({-1.0, 0.0, 1.0})
    assert alpha > 0


def test_ste_passes_gradients():
    w = torch.randn(8, 8, requires_grad=True)
    out = ste_ternarize(w).sum()
    out.backward()
    assert w.grad is not None
    assert torch.all(w.grad == 1.0)  # STE => identity gradient


def test_bitlinear_forward_backward():
    layer = BitLinear(16, 8)
    x = torch.randn(4, 16, requires_grad=True)
    y = layer(x)
    assert y.shape == (4, 8)
    y.sum().backward()
    assert layer.weight.grad is not None
    assert layer.weight.grad.abs().sum() > 0


def test_bitlinear_weights_remain_trainable():
    layer = BitLinear(8, 4)
    opt = torch.optim.SGD(layer.parameters(), lr=0.1)
    w_before = layer.weight.detach().clone()
    x = torch.randn(32, 8)
    target = torch.randn(32, 4)
    for _ in range(5):
        opt.zero_grad()
        loss = ((layer(x) - target) ** 2).mean()
        loss.backward()
        opt.step()
    assert not torch.allclose(w_before, layer.weight)


def test_ternary_stats_sum_to_one():
    layer = BitLinear(32, 16)
    s = layer.ternary_stats()
    assert abs(s["neg"] + s["zero"] + s["pos"] - 1.0) < 1e-6
