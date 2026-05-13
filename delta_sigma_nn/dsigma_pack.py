"""Pack and unpack delta-sigma trit streams for deployment.

A trained DeltaSigmaMLP has a target-weight tensor per BitLinear-like layer.
At inference we don't want to re-run the modulator every forward; instead
we pre-compute the T-step stream once and store it as packed bytes.

Storage per layer:
  - packed stream:  shape (T, out, in), trits → 5 trits/byte → ~0.2*T*out*in bytes
  - alpha (per-layer scale): 4 bytes
  - LayerNorm gamma, beta: 4 * in bytes each (small)
  - bias: 4 * out bytes

vs fp32:
  - W: 4 * out * in bytes

For T=8: dsigma cost = 0.2 * 8 = 1.6 bytes/weight (same as 1 trit + 8/5 oversample)
For T=4: 0.8 bytes/weight (vs 4 for fp32 → 5x compression)

The deployment advantage: with the packed stream we can run *anytime*
inference at inference time — the streams give us a runtime knob.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.special import erf as _erf

from .delta_sigma import encode_delta_sigma_order2, encode_delta_sigma_ternary
from .dsigma_linear import DeltaSigmaLinear, DeltaSigmaMLP
from .trit_pack import pack_trits, unpack_trits


def pack_dsigma_layer(layer: DeltaSigmaLinear) -> dict[str, np.ndarray]:
    """Pre-encode a layer's stream and pack it."""
    with torch.no_grad():
        alpha = layer.weight.abs().mean().clamp_min(1e-5)
        W_norm = (layer.weight / alpha).clamp(-1.0, 1.0)
        if layer.order == 1:
            stream = encode_delta_sigma_ternary(W_norm, T=layer.T)
        else:
            stream = encode_delta_sigma_order2(W_norm, T=layer.T)

    T, out_dim, in_dim = stream.shape
    # Flatten time dimension into the weight matrix so we have one packed
    # blob per layer instead of T smaller ones.
    flat = stream.numpy().astype(np.int8).reshape(-1)  # (T * out * in,)
    packed, n_trits = pack_trits(flat)

    return {
        "packed_stream": packed,                                   # uint8
        "stream_shape": np.array([T, out_dim, in_dim], dtype=np.int64),
        "alpha": np.array([float(alpha)], dtype=np.float32),
        "bias": (layer.bias.detach().numpy().astype(np.float32)
                 if layer.bias is not None else np.zeros(0, dtype=np.float32)),
        "ln_weight": layer.norm.weight.detach().numpy().astype(np.float32),
        "ln_bias": layer.norm.bias.detach().numpy().astype(np.float32),
        "T": np.array([T], dtype=np.int64),
    }


def unpack_dsigma_layer(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Inverse: load and reconstruct the stream."""
    T, out_dim, in_dim = data["stream_shape"]
    n_trits = T * out_dim * in_dim
    flat = unpack_trits(data["packed_stream"], n_trits)
    stream = flat.reshape(T, out_dim, in_dim)
    return {
        "stream": stream,                                       # int8 in {-1,0,1}
        "alpha": float(data["alpha"][0]),
        "bias": data["bias"],
        "ln_weight": data["ln_weight"],
        "ln_bias": data["ln_bias"],
        "T": int(data["T"][0]),
    }


def save_dsigma_mlp(model: DeltaSigmaMLP, path: str | Path) -> dict[str, int]:
    """Serialize a trained DeltaSigmaMLP to disk."""
    path = Path(path)
    blobs: dict[str, np.ndarray] = {}

    dsigmas = list(model.dsigma_blocks)

    blobs["boundary_in_W"]  = model.in_proj.weight.detach().numpy().astype(np.float32)
    blobs["boundary_in_b"]  = model.in_proj.bias.detach().numpy().astype(np.float32)
    blobs["boundary_out_W"] = model.out_proj.weight.detach().numpy().astype(np.float32)
    blobs["boundary_out_b"] = model.out_proj.bias.detach().numpy().astype(np.float32)
    blobs["layer_count"] = np.array(len(dsigmas), dtype=np.int64)

    breakdown = {
        "boundary_in_W_bytes":  blobs["boundary_in_W"].nbytes,
        "boundary_out_W_bytes": blobs["boundary_out_W"].nbytes,
        "dsigma_layers_bytes": 0,
    }

    for i, layer in enumerate(dsigmas):
        packed_data = pack_dsigma_layer(layer)
        for k, v in packed_data.items():
            blobs[f"ds_{i}_{k}"] = v
        breakdown["dsigma_layers_bytes"] += packed_data["packed_stream"].nbytes
        breakdown[f"layer_{i}_stream_bytes"] = packed_data["packed_stream"].nbytes

    np.savez(path, **blobs)
    breakdown["file_bytes_on_disk"] = path.stat().st_size
    return breakdown


def load_dsigma_arrays(path: str | Path) -> dict:
    """Load packed model into a plain dict of arrays."""
    data = np.load(path)
    n = int(data["layer_count"])
    out: dict = {
        "boundary_in_W":  data["boundary_in_W"],
        "boundary_in_b":  data["boundary_in_b"],
        "boundary_out_W": data["boundary_out_W"],
        "boundary_out_b": data["boundary_out_b"],
        "layers": [],
    }
    for i in range(n):
        layer_data = {
            "packed_stream": data[f"ds_{i}_packed_stream"],
            "stream_shape":  data[f"ds_{i}_stream_shape"],
            "alpha":         data[f"ds_{i}_alpha"],
            "bias":          data[f"ds_{i}_bias"],
            "ln_weight":     data[f"ds_{i}_ln_weight"],
            "ln_bias":       data[f"ds_{i}_ln_bias"],
            "T":             data[f"ds_{i}_T"],
        }
        out["layers"].append(unpack_dsigma_layer(layer_data))
    return out


def dsigma_inference(arrays: dict, x: np.ndarray, k: int | None = None) -> np.ndarray:
    """Pure-NumPy forward pass using packed delta-sigma streams.

    If k is None, use full T. Otherwise truncate to first k time steps
    (anytime inference).

    This is a *correctness oracle* for the multiply-free claim, not a
    multiply-free implementation: we fold the T trit slices into a single
    fp32 effective matrix and call `@`, which does use floating-point
    multiplications. The multiply-free path is the Verilog backend in
    `hardware/` (or any T-step accumulate-and-shift implementation). For
    PyTorch/NumPy this fused form is faster and bit-identical at full T.
    """
    def layernorm(x, w, b, eps=1e-5):
        mu = x.mean(axis=-1, keepdims=True)
        var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
        return ((x - mu) / np.sqrt(var + eps)) * w + b

    def gelu(x):
        return 0.5 * x * (1.0 + _erf(x / np.sqrt(2.0)))

    def fp_linear(x, W, b):
        return x @ W.T + b

    h = x.astype(np.float32)
    if h.ndim == 1:
        h = h[None, :]
        squeeze = True
    else:
        squeeze = False

    h = fp_linear(h, arrays["boundary_in_W"], arrays["boundary_in_b"])
    h = gelu(h)

    for layer in arrays["layers"]:
        h_norm = layernorm(h, layer["ln_weight"], layer["ln_bias"])
        stream = layer["stream"]  # (T, out, in)
        T = layer["T"]
        k_eff = T if k is None else min(k, T)
        # Average the first k_eff time slices, then scale by alpha.
        # The summation across time and the dot product are interchangeable.
        w_eff = stream[:k_eff].mean(axis=0) * layer["alpha"]
        # Cast int-derived w to float32 for the matmul
        h = h_norm @ w_eff.astype(np.float32).T
        if layer["bias"].size > 0:
            h = h + layer["bias"]
        h = gelu(h)

    h = fp_linear(h, arrays["boundary_out_W"], arrays["boundary_out_b"])
    return h.squeeze(0) if squeeze else h
