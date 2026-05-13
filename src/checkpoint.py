"""Save/load a trained BitMLP with hidden weights as packed trits.

Format (numpy .npz):
    layer_count: int
    boundary_in_W, boundary_in_b      # fp32 input projection (Linear)
    boundary_out_W, boundary_out_b    # fp32 output head (Linear)
    For each ternary hidden layer i:
        bit_{i}_packed   uint8   (trit-packed weights)
        bit_{i}_shape    int64   (out_features, in_features)
        bit_{i}_alpha    float32 (scalar scale)
        bit_{i}_bias     float32 (out_features,) or empty
        bit_{i}_ln_w     float32 (in_features,)
        bit_{i}_ln_b     float32 (in_features,)
    norm_stats_mean, norm_stats_std   (input normalization)

This is what an actual on-device deployment of a BitNet b1.58 model
would store: bulk weights at ~1.6 bits/weight, tiny boundary tensors
at fp32, no fp32 matrices except the (small) boundary projections.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .model import BitMLP
from .ternary import BitLinear, ternarize
from delta_sigma_nn.trit_pack import pack_trits, unpack_trits, storage_bytes


def save_bitmlp(model: BitMLP, path: str | Path,
                norm_stats: dict | None = None) -> dict[str, int]:
    """Save a trained BitMLP. Returns byte-size breakdown of the saved file."""
    path = Path(path)
    blobs: dict[str, np.ndarray] = {}

    # Walk the net in order, collecting Linear and BitLinear layers.
    linears = []
    bits = []
    for m in model.net:
        if isinstance(m, BitLinear):
            bits.append(m)
        elif isinstance(m, torch.nn.Linear):
            linears.append(m)

    assert len(linears) == 2, "expected exactly 2 fp32 boundary Linear layers"
    blobs["boundary_in_W"] = linears[0].weight.detach().numpy().astype(np.float32)
    blobs["boundary_in_b"] = linears[0].bias.detach().numpy().astype(np.float32)
    blobs["boundary_out_W"] = linears[1].weight.detach().numpy().astype(np.float32)
    blobs["boundary_out_b"] = linears[1].bias.detach().numpy().astype(np.float32)

    blobs["layer_count"] = np.array(len(bits), dtype=np.int64)

    breakdown = {
        "boundary_in_W_bytes": blobs["boundary_in_W"].nbytes,
        "boundary_in_b_bytes": blobs["boundary_in_b"].nbytes,
        "boundary_out_W_bytes": blobs["boundary_out_W"].nbytes,
        "boundary_out_b_bytes": blobs["boundary_out_b"].nbytes,
        "ternary_layers_bytes": 0,
    }

    for i, layer in enumerate(bits):
        with torch.no_grad():
            w_q, alpha = ternarize(layer.weight)
            w_int = w_q.numpy().astype(np.int8)
        packed, _ = pack_trits(w_int)
        out_dim, in_dim = w_int.shape
        blobs[f"bit_{i}_packed"] = packed
        blobs[f"bit_{i}_shape"] = np.array([out_dim, in_dim], dtype=np.int64)
        blobs[f"bit_{i}_alpha"] = np.array([float(alpha)], dtype=np.float32)
        if layer.bias is not None:
            blobs[f"bit_{i}_bias"] = layer.bias.detach().numpy().astype(np.float32)
        else:
            blobs[f"bit_{i}_bias"] = np.zeros(0, dtype=np.float32)
        blobs[f"bit_{i}_ln_w"] = layer.norm.weight.detach().numpy().astype(np.float32)
        blobs[f"bit_{i}_ln_b"] = layer.norm.bias.detach().numpy().astype(np.float32)

        layer_bytes = (
            packed.nbytes
            + 4  # alpha
            + blobs[f"bit_{i}_bias"].nbytes
            + blobs[f"bit_{i}_ln_w"].nbytes
            + blobs[f"bit_{i}_ln_b"].nbytes
        )
        breakdown["ternary_layers_bytes"] += layer_bytes
        breakdown[f"layer_{i}_packed_bytes"] = packed.nbytes
        breakdown[f"layer_{i}_total_bytes"] = layer_bytes

    if norm_stats is not None:
        blobs["norm_stats_mean"] = norm_stats["mean"].numpy().astype(np.float32)
        blobs["norm_stats_std"] = norm_stats["std"].numpy().astype(np.float32)

    np.savez(path, **blobs)
    breakdown["file_bytes_on_disk"] = path.stat().st_size
    return breakdown


def load_bitmlp_arrays(path: str | Path) -> dict:
    """Load a saved BitMLP as a dict of numpy arrays (ready for multfree inference)."""
    data = np.load(path)
    out: dict = {
        "boundary_in_W": data["boundary_in_W"],
        "boundary_in_b": data["boundary_in_b"],
        "boundary_out_W": data["boundary_out_W"],
        "boundary_out_b": data["boundary_out_b"],
        "layers": [],
    }
    n = int(data["layer_count"])
    for i in range(n):
        packed = data[f"bit_{i}_packed"]
        shape = tuple(data[f"bit_{i}_shape"].tolist())
        n_trits = shape[0] * shape[1]
        W = unpack_trits(packed, n_trits).reshape(shape)
        layer = {
            "W": W,                              # int8 in {-1, 0, +1}
            "alpha": float(data[f"bit_{i}_alpha"][0]),
            "bias": data[f"bit_{i}_bias"],       # may be empty
            "ln_w": data[f"bit_{i}_ln_w"],
            "ln_b": data[f"bit_{i}_ln_b"],
            "shape": shape,
        }
        out["layers"].append(layer)
    if "norm_stats_mean" in data.files:
        out["norm_stats_mean"] = data["norm_stats_mean"]
        out["norm_stats_std"] = data["norm_stats_std"]
    return out
