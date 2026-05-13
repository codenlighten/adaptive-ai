import torch

from src.quintary import BinaryLinear, QuintLinear, binarize, quintize


def test_quintize_values_in_set():
    torch.manual_seed(0)
    w = torch.randn(64, 64) * 0.1
    w_q, alpha = quintize(w)
    unique = torch.unique(w_q).tolist()
    assert set(unique).issubset({-2.0, -1.0, 0.0, 1.0, 2.0})
    assert alpha > 0


def test_binarize_values_in_set():
    w = torch.randn(64, 64)
    w_q, _ = binarize(w)
    assert set(torch.unique(w_q).tolist()).issubset({-1.0, 1.0})


def test_quintlinear_trains():
    torch.manual_seed(0)
    layer = QuintLinear(8, 4)
    opt = torch.optim.SGD(layer.parameters(), lr=0.05)
    w_before = layer.weight.detach().clone()
    x = torch.randn(32, 8)
    target = torch.randn(32, 4)
    for _ in range(5):
        opt.zero_grad()
        loss = ((layer(x) - target) ** 2).mean()
        loss.backward()
        opt.step()
    assert not torch.allclose(w_before, layer.weight)


def test_binarylinear_forward():
    layer = BinaryLinear(8, 4)
    out = layer(torch.randn(2, 8))
    assert out.shape == (2, 4)
