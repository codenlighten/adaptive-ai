"""Pack balanced ternary {-1, 0, +1} arrays into bytes.

5 trits fit in 1 byte because 3^5 = 243 < 256. Each trit is mapped:
    -1 -> 0,  0 -> 1,  +1 -> 2
then 5 trits (t0..t4) become a base-3 digit string:
    byte = t0 + 3*t1 + 9*t2 + 27*t3 + 81*t4

Storage ratio vs fp32 weights: 32 bits / (8/5 bits) = 20x compression.
Vs int8: 8 bits / 1.6 bits = 5x compression.
"""

from __future__ import annotations

import numpy as np

_POW3 = np.array([1, 3, 9, 27, 81], dtype=np.uint16)


def pack_trits(arr: np.ndarray) -> tuple[np.ndarray, int]:
    """Pack an array of values in {-1, 0, +1} into bytes (5 trits / byte).

    Returns (packed_bytes, original_length). The length is needed for unpack
    because the final byte may encode fewer than 5 trits.
    """
    flat = np.asarray(arr, dtype=np.int8).ravel()
    if flat.size == 0:
        return np.zeros(0, dtype=np.uint8), 0
    assert flat.min() >= -1 and flat.max() <= 1, "values must be in {-1, 0, 1}"

    n = flat.size
    pad = (-n) % 5
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.int8)])
    shifted = (flat + 1).astype(np.uint16).reshape(-1, 5)  # {-1,0,1} -> {0,1,2}
    packed = (shifted * _POW3).sum(axis=1).astype(np.uint8)
    return packed, n


def unpack_trits(packed: np.ndarray, length: int) -> np.ndarray:
    """Inverse of pack_trits. Returns an int8 array of `length` trits."""
    if length == 0:
        return np.zeros(0, dtype=np.int8)
    vals = packed.astype(np.uint16)
    out = np.zeros((vals.size, 5), dtype=np.int8)
    for i in range(5):
        out[:, i] = (vals % 3).astype(np.int8) - 1
        vals //= 3
    return out.ravel()[:length]


def storage_bytes(n_trits: int) -> int:
    """How many bytes the packed representation occupies."""
    return (n_trits + 4) // 5


def compression_ratio_vs_fp32(n_trits: int) -> float:
    return (n_trits * 4) / storage_bytes(n_trits)


def compression_ratio_vs_int8(n_trits: int) -> float:
    return n_trits / storage_bytes(n_trits)
