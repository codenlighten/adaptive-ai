import torch

from src.stochastic_trit import StochasticTernaryActivation, stochastic_trit


def test_outputs_are_trits_during_sampling():
    torch.manual_seed(0)
    x = torch.randn(8, 32)
    y = stochastic_trit(x)
    assert set(torch.unique(y).tolist()).issubset({-1.0, 0.0, 1.0})


def test_expectation_is_unbiased():
    """Average of many samples should approach the clipped input."""
    torch.manual_seed(0)
    x = torch.linspace(-1.0, 1.0, 7)
    n = 5000
    samples = torch.stack([stochastic_trit(x) for _ in range(n)])
    mean = samples.mean(0)
    # The expectation of stochastic_trit is clamp(x, -1, 1) — for x in [-1,1] that's x itself.
    assert torch.allclose(mean, x, atol=0.05), f"{mean.tolist()} vs {x.tolist()}"


def test_saturation():
    """Inputs > 1 should always produce +1, < -1 always -1."""
    torch.manual_seed(0)
    x = torch.tensor([3.0, -3.0, 0.0])
    n = 200
    for _ in range(n):
        y = stochastic_trit(x)
        assert y[0] == 1.0
        assert y[1] == -1.0
        assert y[2] == 0.0


def test_ste_gradient_passes():
    torch.manual_seed(0)
    x = torch.randn(16, requires_grad=True)
    y = stochastic_trit(x).sum()
    y.backward()
    assert x.grad is not None
    assert torch.allclose(x.grad, torch.ones_like(x))


def test_activation_module_train_eval_modes():
    act = StochasticTernaryActivation()
    x = torch.linspace(-2, 2, 64).unsqueeze(0)
    act.eval()
    y_eval = act(x)
    # eval mode: deterministic (clamped) — same output on repeated calls
    y_eval2 = act(x)
    assert torch.allclose(y_eval, y_eval2)
    act.train()
    torch.manual_seed(0)
    y_train1 = act(x)
    torch.manual_seed(1)
    y_train2 = act(x)
    assert not torch.allclose(y_train1, y_train2)
