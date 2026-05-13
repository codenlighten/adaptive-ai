import threading

import torch

from delta_sigma_nn.dsigma_linear import DeltaSigmaLinear, dsigma_inference_context
from delta_sigma_nn.dsigma_router import confidence_router
from delta_sigma_nn.dsigma_transformer import DSigmaCharLM


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


def test_router_does_not_mutate_class():
    """confidence_router must not monkey-patch DeltaSigmaLinear.forward."""
    original_forward = DeltaSigmaLinear.forward
    torch.manual_seed(0)
    model = DSigmaCharLM(32, d_model=32, n_heads=2, n_layers=2, max_len=8, T=8)
    model.eval()
    idx = torch.randint(0, 32, (1, 6))
    confidence_router(model, idx, k_schedule=[1, 2, 4], signal="diff", threshold=10.0)
    assert DeltaSigmaLinear.forward is original_forward


def test_router_clears_instance_caches():
    """After the router returns, no DeltaSigmaLinear should retain cached state."""
    torch.manual_seed(0)
    model = DSigmaCharLM(32, d_model=32, n_heads=2, n_layers=2, max_len=8, T=8)
    model.eval()
    idx = torch.randint(0, 32, (1, 6))
    confidence_router(model, idx, k_schedule=[1, 2], signal="diff", threshold=10.0)
    for m in model.modules():
        if isinstance(m, DeltaSigmaLinear):
            for attr in ("_cached_stream", "_cached_alpha", "_truncation_k"):
                assert not hasattr(m, attr), f"leaked {attr} on {m}"


def test_inference_context_round_trip():
    """The context manager should not change forward results outside its scope."""
    torch.manual_seed(0)
    model = DSigmaCharLM(32, d_model=32, n_heads=2, n_layers=2, max_len=8, T=8)
    model.eval()
    idx = torch.randint(0, 32, (1, 6))
    with torch.no_grad():
        before = model(idx)
        with dsigma_inference_context(model, k=8):  # full T
            inside = model(idx)
        after = model(idx)
    # Full-k inference should match the encode-each-call path within float noise.
    assert (inside - before).abs().max().item() < 1e-5
    assert (after - before).abs().max().item() < 1e-5


def test_routers_run_concurrently_on_separate_models():
    """Two independent models in two threads must not corrupt each other."""
    torch.manual_seed(0)
    model_a = DSigmaCharLM(32, d_model=32, n_heads=2, n_layers=2, max_len=8, T=8)
    model_b = DSigmaCharLM(32, d_model=32, n_heads=2, n_layers=2, max_len=8, T=8)
    model_a.eval()
    model_b.eval()
    idx = torch.randint(0, 32, (1, 6))
    results: dict[str, tuple] = {}
    errors: list[BaseException] = []

    def run(name, model):
        try:
            for _ in range(3):
                logits, k = confidence_router(model, idx, k_schedule=[1, 2, 4, 8],
                                              signal="diff", threshold=0.1)
                results[name] = (logits, k)
        except BaseException as exc:  # noqa: BLE001 — re-raise after join
            errors.append(exc)

    t1 = threading.Thread(target=run, args=("a", model_a))
    t2 = threading.Thread(target=run, args=("b", model_b))
    t1.start(); t2.start(); t1.join(); t2.join()
    assert not errors, errors
    assert "a" in results and "b" in results
    # And the class method is still pristine after concurrent use.
    for m in list(model_a.modules()) + list(model_b.modules()):
        if isinstance(m, DeltaSigmaLinear):
            assert not hasattr(m, "_cached_stream")
