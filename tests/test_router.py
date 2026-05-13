import torch

from src.dsigma_router import confidence_router
from src.dsigma_transformer import DSigmaCharLM


def test_router_returns_logits():
    torch.manual_seed(0)
    model = DSigmaCharLM(64, d_model=32, n_heads=2, n_layers=2, max_len=16, T=8)
    model.eval()
    idx = torch.randint(0, 64, (2, 10))
    logits, k = confidence_router(model, idx, k_schedule=[1, 2, 4, 8],
                                   signal="diff", threshold=10.0)
    assert logits.shape == (2, 10, 64)
    assert k in [1, 2, 4, 8]


def test_router_full_k_at_strict_threshold():
    """A very strict threshold should force full k."""
    torch.manual_seed(0)
    model = DSigmaCharLM(32, d_model=32, n_heads=2, n_layers=2, max_len=8, T=8)
    model.eval()
    idx = torch.randint(0, 32, (1, 6))
    _, k = confidence_router(model, idx, k_schedule=[1, 2, 4, 8],
                              signal="diff", threshold=1e-12)
    assert k == 8


def test_router_low_k_at_loose_threshold():
    """A very loose threshold should stop early."""
    torch.manual_seed(0)
    model = DSigmaCharLM(32, d_model=32, n_heads=2, n_layers=2, max_len=8, T=8)
    model.eval()
    idx = torch.randint(0, 32, (1, 6))
    _, k = confidence_router(model, idx, k_schedule=[1, 2, 4, 8],
                              signal="diff", threshold=100.0)
    assert k == 2  # second iter triggers diff signal


def test_three_signals_run():
    torch.manual_seed(0)
    model = DSigmaCharLM(32, d_model=32, n_heads=2, n_layers=2, max_len=8, T=8)
    model.eval()
    idx = torch.randint(0, 32, (1, 6))
    for sig, thr in [("entropy", 1.0), ("topk_gap", 0.5), ("diff", 0.1)]:
        _, k = confidence_router(model, idx, signal=sig, threshold=thr)
        assert k > 0
