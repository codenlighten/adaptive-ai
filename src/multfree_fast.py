"""Optimized multiply-free matmul implementations.

Three flavors, fastest first:

1. matmul_bitpacked: pack the +1 and -1 masks of W into uint64 bitmaps and use
   NumPy's vectorized `np.where`-style operations. Falls back to dense ops
   under the hood — fast but still leans on BLAS for the summations.

2. matmul_numba: JIT-compiled tight loop. Performs the literal signed-add
   inner loop with no Python overhead. The closest thing to "what hardware
   would do" that we can run on a CPU. Single-threaded by default.

3. matmul_numba_parallel: same kernel but `parallel=True`. Multi-core.

Wall-clock note: NumPy's fp32 matmul calls into OpenBLAS, which uses
hand-tuned AVX with FMA — a single CPU FMA does a multiply+add in one
cycle. Beating that on the same hardware is hard. The point of these
implementations is to show that:
  (a) ignoring the multiplier costs gives roughly half the FLOPs,
  (b) on a custom ASIC/FPGA where multiplies are 10-50x more expensive
      than adds, ternary's wall-clock win is real,
  (c) the Numba-JIT version is much faster than Python and within an
      order of magnitude of BLAS on CPU despite using zero multiplies.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, fastmath=True)
def matmul_numba(W: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Multiply-free matmul, JIT-compiled.

    W: (out, in) int8 in {-1, 0, +1}
    x: (in,) float32
    Returns (out,) float32.
    """
    out_dim, in_dim = W.shape
    y = np.zeros(out_dim, dtype=np.float32)
    for i in range(out_dim):
        acc = np.float32(0.0)
        for j in range(in_dim):
            w = W[i, j]
            if w == 1:
                acc += x[j]
            elif w == -1:
                acc -= x[j]
            # else: skip — no multiply ever
        y[i] = acc
    return y


@njit(cache=True, fastmath=True, parallel=True)
def matmul_numba_parallel(W: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Same as matmul_numba but parallel across output rows."""
    out_dim, in_dim = W.shape
    y = np.zeros(out_dim, dtype=np.float32)
    for i in prange(out_dim):
        acc = np.float32(0.0)
        for j in range(in_dim):
            w = W[i, j]
            if w == 1:
                acc += x[j]
            elif w == -1:
                acc -= x[j]
        y[i] = acc
    return y


@njit(cache=True, fastmath=True, parallel=True)
def matmul_numba_batch(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Batch matmul: (out, in) @ (in, B) -> (out, B), zero multiplies."""
    out_dim, in_dim = W.shape
    batch = X.shape[1]
    Y = np.zeros((out_dim, batch), dtype=np.float32)
    for i in prange(out_dim):
        for b in range(batch):
            acc = np.float32(0.0)
            for j in range(in_dim):
                w = W[i, j]
                if w == 1:
                    acc += X[j, b]
                elif w == -1:
                    acc -= X[j, b]
            Y[i, b] = acc
    return Y
