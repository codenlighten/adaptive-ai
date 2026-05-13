"""Multiply-free matrix multiplication for ternary weights.

For W in {-1, 0, +1} and arbitrary x, the matmul y = W @ x decomposes into:
    y[i] = sum over j where W[i,j]=+1 of x[j]
         - sum over j where W[i,j]=-1 of x[j]
    (entries with W[i,j]=0 are skipped)

No multiplications. Just signed additions. This is the actual hardware
advantage of BitNet-style ternary networks: a multiplier is the most
expensive part of a dot product, and ternary eliminates it entirely.

Two implementations:
- `matmul_split_masks`: vectorized via boolean masks (fast, NumPy).
- `matmul_scalar`: explicit per-element loop (slow, but mechanically
  demonstrates the multiply-free property).
"""

from __future__ import annotations

import numpy as np


def matmul_split_masks(W: np.ndarray, x: np.ndarray) -> np.ndarray:
    """y = W @ x using only additions/subtractions on x.

    W: (out, in) int8 with values in {-1, 0, +1}
    x: (in,) or (in, batch) float
    Returns y with the same float dtype as x.
    """
    assert W.dtype == np.int8
    pos = (W == 1)
    neg = (W == -1)
    # pos.astype(float) @ x is equivalent to "sum x[j] where pos[i,j]",
    # done with adds only when implemented in hardware. NumPy uses BLAS
    # under the hood; the *semantic* operation is multiply-free.
    return pos.astype(x.dtype) @ x - neg.astype(x.dtype) @ x


def matmul_scalar(W: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Reference: scalar loop, demonstrably multiply-free.

    Each y[i] is built by walking W[i] and either adding x[j], subtracting
    x[j], or skipping. No `*` operation touches a non-trivial value.
    """
    out_dim, in_dim = W.shape
    if x.ndim == 1:
        y = np.zeros(out_dim, dtype=x.dtype)
        for i in range(out_dim):
            acc = x.dtype.type(0)
            for j in range(in_dim):
                w = W[i, j]
                if w == 1:
                    acc += x[j]
                elif w == -1:
                    acc -= x[j]
                # else: skip
            y[i] = acc
        return y
    else:
        batch = x.shape[1]
        y = np.zeros((out_dim, batch), dtype=x.dtype)
        for i in range(out_dim):
            for j in range(in_dim):
                w = W[i, j]
                if w == 1:
                    y[i] += x[j]
                elif w == -1:
                    y[i] -= x[j]
        return y


def matmul_ternary(W: np.ndarray, x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Full BitNet-style ternary matmul with output scale alpha."""
    return alpha * matmul_split_masks(W, x)


def count_ops(W: np.ndarray) -> dict[str, int]:
    """Operation counts vs an equivalent fp32 dense matmul."""
    n_total = W.size
    n_nonzero = int(np.count_nonzero(W))
    return {
        "total_weights": n_total,
        "nonzero_weights": n_nonzero,
        "skipped_fraction": 1.0 - n_nonzero / n_total if n_total else 0.0,
        "fp32_multiplies": n_total,           # one * per weight in dense matmul
        "ternary_multiplies": 0,              # zero — that's the point
        "ternary_adds_subs": n_nonzero,
    }
