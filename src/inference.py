"""Pure-NumPy inference engine for trained BitMLPs.

Loads a checkpoint saved by checkpoint.save_bitmlp. The hidden-layer matmuls
go through multfree.matmul_split_masks — semantically zero floating-point
multiplies per weight (only signed additions). A FLOPCounter wraps every
fp multiplication so we can verify the claim numerically.

What's still multiplicative (and unavoidable for the b1.58 recipe):
  - the per-layer scalar `alpha * y` rescale (1 multiply per output element)
  - LayerNorm's affine scale + 1/sqrt
  - GELU
  - the fp32 boundary input/output projections (small)
All of these are O(width) or constant per output, not O(width^2) like a matmul.
The big-O multiplier savings is in the matmul itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .checkpoint import load_bitmlp_arrays
from .multfree import matmul_split_masks


@dataclass
class OpCounter:
    """Track every fp multiplication that happens during inference."""

    matmul_multiplies_avoided: int = 0   # would-be fp multiplies in matmuls
    matmul_signed_adds: int = 0          # signed adds we actually do
    scalar_multiplies: int = 0           # alpha * y, layernorm scale, etc.
    fp_boundary_multiplies: int = 0      # full-precision matmuls (small)
    other_multiplies: int = 0            # GELU, etc.

    def total_real_multiplies(self) -> int:
        return self.scalar_multiplies + self.fp_boundary_multiplies + self.other_multiplies


def gelu(x: np.ndarray, counter: OpCounter) -> np.ndarray:
    # Exact GELU (matches torch.nn.GELU default): 0.5 * x * (1 + erf(x / sqrt(2)))
    from math import sqrt
    from scipy.special import erf as _erf
    counter.other_multiplies += x.size * 2  # the two mults in 0.5 * x * (...)
    return 0.5 * x * (1.0 + _erf(x / sqrt(2.0)))


def layernorm(x: np.ndarray, w: np.ndarray, b: np.ndarray,
              counter: OpCounter, eps: float = 1e-5) -> np.ndarray:
    # mean/var across last axis; then (x-mu)/std * w + b
    mu = x.mean(axis=-1, keepdims=True)
    centered = x - mu
    var = (centered ** 2).mean(axis=-1, keepdims=True)
    counter.scalar_multiplies += centered.size  # centered ** 2
    inv = 1.0 / np.sqrt(var + eps)
    counter.scalar_multiplies += inv.size       # one 1/sqrt per row
    normalized = centered * inv
    counter.scalar_multiplies += normalized.size
    out = normalized * w + b
    counter.scalar_multiplies += normalized.size
    return out


def fp_linear(x: np.ndarray, W: np.ndarray, b: np.ndarray,
              counter: OpCounter) -> np.ndarray:
    # Real fp matmul on the boundary projection. We charge full multiplies.
    counter.fp_boundary_multiplies += x.shape[0] * W.shape[0] * W.shape[1] if x.ndim == 2 \
        else W.shape[0] * W.shape[1]
    return x @ W.T + b


def bit_linear(x: np.ndarray, layer: dict, counter: OpCounter) -> np.ndarray:
    """One BitLinear forward pass with zero multiplies in the matmul itself."""
    # LayerNorm first (BitNet recipe)
    x = layernorm(x, layer["ln_w"], layer["ln_b"], counter)

    W = layer["W"]                          # (out, in) int8 in {-1, 0, +1}
    out_dim, in_dim = W.shape

    # Multiply-free matmul: y = W @ x  using only adds/subs
    # x can be (B, in) — handle both 1-D and 2-D.
    if x.ndim == 1:
        # split_masks expects (in,) -> returns (out,)
        y = matmul_split_masks(W, x)
        n_batch = 1
    else:
        # x is (B, in). split_masks does (out, in) @ (in, B) -> (out, B)
        y = matmul_split_masks(W, x.T).T
        n_batch = x.shape[0]

    counter.matmul_multiplies_avoided += n_batch * out_dim * in_dim
    counter.matmul_signed_adds += int(np.count_nonzero(W)) * n_batch

    # Per-output scalar rescale: alpha * y. This is one fp multiply per output.
    y = layer["alpha"] * y
    counter.scalar_multiplies += y.size

    if layer["bias"].size > 0:
        y = y + layer["bias"]
    return y


class TernaryNet:
    """Pure-NumPy inference for a saved BitMLP. Multiply-free in matmuls."""

    def __init__(self, path: str | Path):
        self.arrays = load_bitmlp_arrays(path)
        self.counter = OpCounter()

    def reset_counter(self) -> None:
        self.counter = OpCounter()

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass. x is (B, in_dim) or (in_dim,) float32."""
        if x.ndim == 1:
            x = x[None, :]
            squeeze = True
        else:
            squeeze = False
        x = x.astype(np.float32)

        # FP boundary: input projection (small Linear, no LN, no scale)
        x = fp_linear(x, self.arrays["boundary_in_W"], self.arrays["boundary_in_b"],
                      self.counter)
        x = gelu(x, self.counter)

        # Ternary hidden layers
        for layer in self.arrays["layers"]:
            x = bit_linear(x, layer, self.counter)
            x = gelu(x, self.counter)

        # FP boundary: output head
        x = fp_linear(x, self.arrays["boundary_out_W"], self.arrays["boundary_out_b"],
                      self.counter)

        return x.squeeze(0) if squeeze else x
