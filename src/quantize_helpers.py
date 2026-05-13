"""Extract quantized weight values from trained models in {Bit, Quat, Quint, Binary}MLP.

Used by the rate-distortion analysis to compute Shannon entropy of the
quantized weight distribution.
"""

from __future__ import annotations

import numpy as np
import torch

from .model import BitMLP
from .quantize_levels import (
    binarize_levels,
    quaternize_levels,
    quintize_levels,
    ternarize_levels,
)


def collect_weights_quantized(model, scheme: str) -> np.ndarray:
    """Return all hidden-layer weights as quantized integer-level codes."""
    quantizer = {
        "binary": binarize_levels,
        "ternary": ternarize_levels,
        "quaternary": quaternize_levels,
        "quintary": quintize_levels,
    }[scheme]
    chunks = []
    with torch.no_grad():
        for m in model.net:
            if hasattr(m, "weight") and hasattr(m, "norm"):
                # Hidden BitLinear / QuintLinear / QuatLinear / BinaryLinear
                chunks.append(quantizer(m.weight.detach()).cpu().numpy().ravel())
    return np.concatenate(chunks) if chunks else np.array([], dtype=int)
