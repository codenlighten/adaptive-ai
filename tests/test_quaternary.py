import torch

from src.quaternary import QuatLinear, QuatMLP, quaternize


def test_quaternize_values_in_set():
    torch.manual_seed(0)
    w = torch.randn(8, 8)
    w_q, alpha = quaternize(w)
    levels = {-3 * alpha.item(), -1 * alpha.item(),
              1 * alpha.item(), 3 * alpha.item()}
    unique = torch.unique(w_q).tolist()
    for u in unique:
        assert any(abs(u - lvl) < 1e-6 for lvl in levels), f"unexpected {u}"


def test_no_zero_level():
    """Quaternary has no zero — every weight should carry signal."""
    torch.manual_seed(0)
    w = torch.randn(16, 16)
    w_q, _ = quaternize(w)
    assert (w_q == 0).sum().item() == 0


def test_quatlinear_trains():
    torch.manual_seed(0)
    layer = QuatLinear(8, 4)
    opt = torch.optim.SGD(layer.parameters(), lr=0.05)
    w0 = layer.weight.detach().clone()
    x = torch.randn(32, 8)
    target = torch.randn(32, 4)
    for _ in range(5):
        opt.zero_grad()
        loss = ((layer(x) - target) ** 2).mean()
        loss.backward()
        opt.step()
    assert not torch.allclose(w0, layer.weight)


def test_quat_stats_sum_to_one():
    layer = QuatLinear(16, 8)
    s = layer.quat_stats()
    fracs = [s[k] for k in s if k.startswith("level_")]
    assert abs(sum(fracs) - 1.0) < 1e-6
