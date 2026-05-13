"""Quantization helpers that return INTEGER-LEVEL codes (not scaled values).

For entropy / storage analysis we need the discrete code per weight,
not the floating-point reconstruction. These mirror the quantize-rules
used elsewhere in the codebase.
"""

from __future__ import annotations

import torch

from .quaternary import quaternize
from .quintary import quintize
from .ternary import ternarize


def ternarize_levels(w: torch.Tensor) -> torch.Tensor:
    w_q, _ = ternarize(w)
    return w_q.to(torch.int8)  # values in {-1, 0, +1}


def quintize_levels(w: torch.Tensor) -> torch.Tensor:
    w_q, _ = quintize(w)
    return w_q.to(torch.int8)  # values in {-2, -1, 0, +1, +2}


def quaternize_levels(w: torch.Tensor) -> torch.Tensor:
    """quaternize returns levels in {-3, -1, +1, +3} * alpha; we restore the int code."""
    w_q, alpha = quaternize(w)
    levels = (w_q / alpha).round().to(torch.int8)
    return levels  # values in {-3, -1, +1, +3}


def binarize_levels(w: torch.Tensor) -> torch.Tensor:
    """Returns sign(w) as int8: values in {-1, +1}."""
    return torch.sign(w).to(torch.int8)
