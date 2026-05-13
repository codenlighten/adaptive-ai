"""Delta-sigma modulation for neural network weights.

Each weight w in [-1, 1] is encoded as a length-T stream of trits
{-1, 0, +1} whose time-average approximates w. We use a first-order
balanced delta-sigma loop:

    integrator = 0
    for t in 1..T:
        integrator += w_target
        if integrator > +theta:     bit = +1
        elif integrator < -theta:   bit = -1
        else:                       bit = 0
        integrator -= bit
        emit bit

The threshold theta controls how aggressively the modulator emits
nonzero bits. With theta = 0.5, a weight of 0 emits a stream of all
zeros; a weight of 1 emits a stream of all +1s; a weight of 0.5
emits {+1, 0, +1, 0, +1, 0, ...}.

The cumulative average of the stream converges to the target as
1/T for first-order delta-sigma (the noise integrates).

This module also provides a binary (1-bit) variant for comparison
with classical sigma-delta modulators.

A weight tensor is encoded element-wise. The matmul below is computed
as the time-average of T one-trit matmuls — each of which uses zero
multiplications.

Closed-form encoding
--------------------
For the default theta=0.5 with no dither, the cumulative sum of bits
emitted up to step t has the closed form

    S_t = sign(t * w) * ceil(|t * w| - 0.5)        (1)

(this is "round half toward zero" of t*w). Bits then follow as
bit_t = S_t - S_{t-1}, so the entire (T, *shape) stream can be built
in a single broadcast multiply + ceil + diff — no Python loop, no
per-step tensor allocation. The training forward only needs the
time-average (= S_T / T), which skips stream materialization entirely.

The closed form is *algebraically* equivalent to the loop. At fp32 it
can differ from the loop by ±1 step in the bit positions because the
loop "resets" after each fire (avoiding drift accumulation) while the
closed form does not — but the time-average is identical, so training
and full-T inference produce the same outputs. Anytime inference at
k<T may see sub-1/k differences in intermediate outputs.

The loop implementation is kept available as a fallback for dither and
non-default theta.
"""

from __future__ import annotations

import torch


def encode_delta_sigma_ternary(W: torch.Tensor, T: int, theta: float = 0.5,
                                noise_dither: float = 0.0) -> torch.Tensor:
    """Encode weights to a (T, *W.shape) tensor of trits.

    Each element of the output is in {-1, 0, +1} and the time-average
    over dim=0 approximates the original W.

    For the default theta=0.5 with no dither this uses a vectorized
    closed form (see module docstring). With dither or non-default theta
    it falls back to the sequential loop, which also supports a small
    random-noise dither so the integrator doesn't get stuck in
    degenerate cycles when w_target == 0.
    """
    if noise_dither == 0.0 and theta == 0.5:
        return _encode_ternary_vec(W, T)
    return _encode_ternary_loop(W, T, theta=theta, noise_dither=noise_dither)


def _encode_ternary_vec(W: torch.Tensor, T: int) -> torch.Tensor:
    """Vectorized closed form for theta=0.5, no dither.

    Builds the (T, *W.shape) stream via a single broadcast + ceil + diff.
    """
    Wc = W.clamp(-1.0, 1.0)
    t = torch.arange(1, T + 1, dtype=W.dtype, device=W.device)
    t = t.view([T] + [1] * Wc.ndim)                          # (T, 1, 1, ...)
    x = t * Wc                                               # (T, *shape)
    S = torch.sign(x) * torch.ceil(x.abs() - 0.5)            # round-half-to-zero
    S_prev = torch.empty_like(S)
    S_prev[0].zero_()
    S_prev[1:] = S[:-1]
    return S - S_prev


def delta_sigma_mean_ternary(W: torch.Tensor, T: int) -> torch.Tensor:
    """Time-average of the ternary delta-sigma stream — no stream allocation.

    Equivalent to `encode_delta_sigma_ternary(W, T).mean(dim=0)` for the
    default theta=0.5, no-dither case, but does not materialize the
    (T, *shape) intermediate. Used in the training forward path.
    """
    Wc = W.clamp(-1.0, 1.0)
    x = T * Wc                                               # (*shape,)
    S_T = torch.sign(x) * torch.ceil(x.abs() - 0.5)
    return S_T / T


def _encode_ternary_loop(W: torch.Tensor, T: int, theta: float = 0.5,
                          noise_dither: float = 0.0) -> torch.Tensor:
    """Reference sequential implementation. Supports dither and any theta."""
    Wc = W.clamp(-1.0, 1.0)
    stream = torch.empty((T,) + W.shape, dtype=W.dtype, device=W.device)
    integrator = torch.zeros_like(Wc)
    for t in range(T):
        integrator = integrator + Wc
        if noise_dither > 0:
            integrator = integrator + noise_dither * (torch.rand_like(Wc) - 0.5)
        bit = torch.where(
            integrator > theta, torch.ones_like(Wc),
            torch.where(integrator < -theta, -torch.ones_like(Wc),
                        torch.zeros_like(Wc))
        )
        integrator = integrator - bit
        stream[t] = bit
    return stream


def decode_delta_sigma(stream: torch.Tensor) -> torch.Tensor:
    """Average the bit-stream over time to recover an approximate weight tensor."""
    return stream.mean(dim=0)


def encode_delta_sigma_binary(W: torch.Tensor, T: int) -> torch.Tensor:
    """Binary (1-bit) delta-sigma: emits only {-1, +1} per time step.

    This is the classical first-order sigma-delta modulator. The output is
    less sparse than the ternary version but each time-step uses exactly
    one signed add per weight.
    """
    Wc = W.clamp(-1.0, 1.0)
    stream = torch.empty((T,) + W.shape, dtype=W.dtype, device=W.device)
    integrator = torch.zeros_like(Wc)
    for t in range(T):
        integrator = integrator + Wc
        bit = torch.sign(integrator)
        # sign(0) == 0 in pytorch but we want ±1; treat 0 as +1 (rare)
        bit = torch.where(bit == 0, torch.ones_like(bit), bit)
        integrator = integrator - bit
        stream[t] = bit
    return stream


def encode_delta_sigma_order2(W: torch.Tensor, T: int, theta: float = 0.5) -> torch.Tensor:
    """Second-order delta-sigma: two integrators in series.

    This shapes the quantization noise more aggressively to high
    "frequencies" (i.e., concentrates the error in the late time steps),
    leaving the running average more accurate at earlier truncation
    points. For anytime inference this means better early stopping.
    """
    Wc = W.clamp(-1.0, 1.0)
    stream = torch.empty((T,) + W.shape, dtype=W.dtype, device=W.device)
    int1 = torch.zeros_like(Wc)
    int2 = torch.zeros_like(Wc)
    for t in range(T):
        int1 = int1 + Wc
        int2 = int2 + int1
        bit = torch.where(
            int2 > theta * 2, torch.ones_like(Wc),
            torch.where(int2 < -theta * 2, -torch.ones_like(Wc),
                        torch.zeros_like(Wc))
        )
        int1 = int1 - bit
        int2 = int2 - bit
        stream[t] = bit
    return stream
