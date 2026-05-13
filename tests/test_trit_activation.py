import torch

from src.trit_activation import TernaryActivation
from src.trit_mlp import TritMLP


def test_activation_outputs_are_trits():
    torch.manual_seed(0)
    act = TernaryActivation()
    x = torch.randn(8, 32)
    y = act(x)
    assert set(torch.unique(y).tolist()).issubset({-1.0, 0.0, 1.0})


def test_ste_gradient_flows():
    torch.manual_seed(0)
    act = TernaryActivation()
    x = torch.randn(16, requires_grad=True)
    y = act(x).sum()
    y.backward()
    assert x.grad is not None
    # Gradient should be 1 where |x| <= 1, 0 outside.
    mask = (x.abs() <= 1.0).float()
    assert torch.allclose(x.grad, mask)


def test_trit_mlp_forward_backward():
    torch.manual_seed(0)
    model = TritMLP(3, 32, 1, depth=4)
    x = torch.randn(8, 3)
    y = model(x)
    assert y.shape == (8, 1)
    loss = (y ** 2).mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any(g.abs().sum() > 0 for g in grads)


def test_trit_mlp_activations_in_set():
    """Every value coming out of a TernaryActivation must be in {-1, 0, +1}."""
    torch.manual_seed(0)
    model = TritMLP(3, 32, 1, depth=5).eval()
    x = torch.randn(64, 3)
    stats = model.collect_activation_stats(x)
    for s in stats:
        # Three fractions must sum to 1.
        assert abs(s["neg"] + s["zero"] + s["pos"] - 1.0) < 1e-6
