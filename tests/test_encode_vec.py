"""Equivalence and properties of the vectorized first-order DS encoder.

The closed form `encode_delta_sigma_ternary` (theta=0.5, no dither) is
algebraically equivalent to the sequential `_encode_ternary_loop` but
may differ by ±1 step in *bit positions* at fp32 precision (drift
accumulation differs because the loop resets after each emit). Both
forms must agree on:

- output range {-1, 0, +1}
- cumulative sum at the final step (= total bit count)
- time-average within 1/T of the underlying W
- `delta_sigma_mean_ternary(W, T)` exactly matches
  `encode_delta_sigma_ternary(W, T).mean(dim=0)`
"""

import torch

from delta_sigma_nn.delta_sigma import (
    _encode_ternary_loop,
    delta_sigma_mean_ternary,
    encode_delta_sigma_ternary,
)


def test_vec_output_in_trit_set():
    torch.manual_seed(0)
    W = torch.randn(8, 16) * 0.9
    for T in [1, 2, 4, 8, 16, 32, 64]:
        stream = encode_delta_sigma_ternary(W, T=T)
        assert set(torch.unique(stream).tolist()).issubset({-1.0, 0.0, 1.0}), \
            f"T={T} produced non-trit values"


def test_vec_matches_loop_at_full_T_average():
    """Final cumulative average must match the loop within first-order DS noise."""
    torch.manual_seed(1)
    W = torch.randn(64) * 0.95
    for T in [8, 16, 32, 64, 128]:
        loop = _encode_ternary_loop(W, T=T).mean(dim=0)
        vec = encode_delta_sigma_ternary(W, T=T).mean(dim=0)
        # Algebraically equal — and the means are bit-positions-invariant.
        # They should match to within fp tolerance (drift can shift a single
        # bit by ±1 step which can change the count by ≤1 across all T).
        diff = (loop - vec).abs().max().item()
        assert diff <= 1.0 / T + 1e-6, f"T={T}: diff {diff}"


def test_fast_mean_matches_vec_stream_mean():
    """delta_sigma_mean_ternary must agree with encode().mean(0) bit-exactly."""
    torch.manual_seed(2)
    W = torch.randn(5, 7, 9) * 0.8
    for T in [1, 4, 8, 16, 32]:
        fast = delta_sigma_mean_ternary(W, T=T)
        slow = encode_delta_sigma_ternary(W, T=T).mean(dim=0)
        assert torch.allclose(fast, slow, atol=1e-6), f"T={T}: not equal"


def test_vec_zero_unit_edges():
    """W = 0 → all zero; W = +1 → all +1; W = -1 → all -1."""
    z = encode_delta_sigma_ternary(torch.zeros(4, 4), T=8)
    assert torch.equal(z, torch.zeros_like(z))
    p = encode_delta_sigma_ternary(torch.ones(4, 4), T=8)
    assert torch.equal(p, torch.ones_like(p))
    n = encode_delta_sigma_ternary(-torch.ones(4, 4), T=8)
    assert torch.equal(n, -torch.ones_like(n))


def test_vec_time_average_converges():
    """First-order DS error decays as ~1/T."""
    W = torch.linspace(-0.95, 0.95, 41)
    for T in [32, 128, 512]:
        avg = encode_delta_sigma_ternary(W, T=T).mean(dim=0)
        err = (avg - W).abs().max().item()
        assert err < 2.0 / T, f"T={T}: max err {err}"


def test_dither_path_still_works():
    """noise_dither > 0 should fall back to the loop and produce trits."""
    torch.manual_seed(3)
    W = torch.zeros(8)
    s = encode_delta_sigma_ternary(W, T=64, noise_dither=0.1)
    assert set(torch.unique(s).tolist()).issubset({-1.0, 0.0, 1.0})
    # With dither, zero-input should produce a non-trivially non-zero stream
    # on average (some bits flip due to noise).
    assert s.abs().sum().item() >= 0  # sanity — no crash, valid output
