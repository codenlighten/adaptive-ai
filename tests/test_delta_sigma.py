import torch

from src.delta_sigma import (
    decode_delta_sigma,
    encode_delta_sigma_binary,
    encode_delta_sigma_order2,
    encode_delta_sigma_ternary,
)


def test_ternary_stream_outputs_are_trits():
    torch.manual_seed(0)
    W = torch.linspace(-1, 1, 9).reshape(3, 3)
    stream = encode_delta_sigma_ternary(W, T=32)
    assert stream.shape == (32, 3, 3)
    unique = torch.unique(stream).tolist()
    assert set(unique).issubset({-1.0, 0.0, 1.0})


def test_time_average_approximates_target():
    """The mean of the stream should approach the target as T grows."""
    W = torch.tensor([-0.7, -0.3, 0.0, 0.3, 0.7])
    for T in [32, 128, 512]:
        stream = encode_delta_sigma_ternary(W, T=T)
        avg = decode_delta_sigma(stream)
        # First-order delta-sigma has error ~ 1/T
        err = (avg - W).abs().max().item()
        assert err < 5.0 / T, f"T={T}: max err {err}"


def test_zero_input_gives_zero_stream():
    """Constant-zero weights should produce all-zero stream (no dither)."""
    W = torch.zeros(4, 4)
    stream = encode_delta_sigma_ternary(W, T=16, theta=0.5, noise_dither=0.0)
    assert torch.allclose(stream, torch.zeros_like(stream))


def test_unit_input_gives_all_ones():
    """w=+1 should produce a stream of all +1s."""
    W = torch.ones(2, 2)
    stream = encode_delta_sigma_ternary(W, T=8, theta=0.5)
    assert torch.allclose(stream, torch.ones_like(stream))


def test_binary_stream_outputs_pm_one():
    W = torch.linspace(-1, 1, 5)
    stream = encode_delta_sigma_binary(W, T=32)
    assert set(torch.unique(stream).tolist()).issubset({-1.0, 1.0})
    # time-average tracks input
    avg = decode_delta_sigma(stream)
    assert (avg - W).abs().max().item() < 0.15


def test_order2_better_early_truncation():
    """Second-order should be more accurate at small T (noise shaping)."""
    torch.manual_seed(0)
    W = torch.rand(64) * 1.6 - 0.8  # in (-0.8, 0.8)
    T_test = 16
    s1 = encode_delta_sigma_ternary(W, T=T_test)
    s2 = encode_delta_sigma_order2(W, T=T_test)
    err1 = (decode_delta_sigma(s1) - W).abs().mean().item()
    err2 = (decode_delta_sigma(s2) - W).abs().mean().item()
    # Order-2 should have lower or comparable mean error at small T.
    # (For ideal noise shaping the order-2 error decays faster.)
    assert err2 <= err1 + 0.05, f"order-2 err {err2} not <= order-1 err {err1}"
