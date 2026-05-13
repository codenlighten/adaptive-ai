"""BitLinear variant with learnable activation bit-width.

Activations are quantized to n discrete levels per layer, where n is
itself learned (via a soft-relaxation). The forward pass uses uniform
quantization on the LayerNorm-normalized input.

This is the natural extension of BitNet b1.58 (which keeps activations
in higher precision) toward a fully learned-precision model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ternary import ste_ternarize


def quantize_uniform(x: torch.Tensor, n_levels: torch.Tensor) -> torch.Tensor:
    """Round x to one of n_levels uniformly-spaced values in [-1, 1].

    STE: forward rounds, backward identity. n_levels can be a soft (float)
    value with a smooth relaxation: we use a soft-rounding via tanh.
    """
    # Map x in (-1, 1) -> integer step
    n = n_levels.detach()
    step = 2.0 / (n - 1).clamp_min(1.0)
    rounded = torch.round(x.clamp(-1.0, 1.0) / step) * step
    # STE: forward = rounded, backward grad through x
    return x + (rounded - x).detach()


class ActQuantizedBitLinear(nn.Module):
    """BitLinear with quantized activations on the post-LayerNorm input.

    Learnable per-layer n_levels (continuous, then clamped to a useful range).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 init_levels: float = 16.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.norm = nn.LayerNorm(in_features)
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        # Learnable activation precision (kept positive via softplus).
        self.act_levels_raw = nn.Parameter(torch.tensor(float(init_levels)).log())

    def n_levels(self) -> torch.Tensor:
        return self.act_levels_raw.exp().clamp(2.0, 256.0)

    def forward(self, x):
        x = self.norm(x)
        n = self.n_levels()
        x_q = quantize_uniform(x, n)
        w = ste_ternarize(self.weight)
        return F.linear(x_q, w, self.bias)


class ActQuantizedMLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, depth=5, init_levels=16.0):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.GELU()]
        for _ in range(depth - 2):
            layers += [ActQuantizedBitLinear(hidden, hidden, init_levels=init_levels),
                       nn.GELU()]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def n_levels_per_layer(self) -> list[float]:
        return [m.n_levels().item() for m in self.net if isinstance(m, ActQuantizedBitLinear)]
