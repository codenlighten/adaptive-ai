"""Equation Learner with ternary weights.

Architecture: instead of generic GELU/Tanh hidden units, the hidden layer
applies a library of explicit basis functions to the inputs and their
products. A ternary "selector" layer (BitLinear, weights in {-1, 0, +1})
then picks which basis functions to combine.

After training, you can read the formula directly off the ternary weights:
each nonzero entry says "+basis_k" or "-basis_k" in the final expression.

For the pendulum, the true H = p^2/2 + (1 - cos q) is exactly representable
as a sum of three basis functions: {1, p^2, cos q}, so a network that can
correctly select these three (and zero out everything else) would recover
the formula perfectly.

This is what ternary's natural-zero state buys us in physics: it's the
sparsity prior that lets a network rediscover known equations.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .ternary import BitLinear, ternarize


# Basis library applied to a (..., 2) input tensor [q, p].
# Each entry is (name, fn(q, p) -> scalar feature).
BASIS_LIBRARY: list[tuple[str, callable]] = [
    ("1",         lambda q, p: torch.ones_like(q)),
    ("q",         lambda q, p: q),
    ("p",         lambda q, p: p),
    ("q^2",       lambda q, p: q * q),
    ("p^2",       lambda q, p: p * p),
    ("q*p",       lambda q, p: q * p),
    ("sin(q)",    lambda q, p: torch.sin(q)),
    ("cos(q)",    lambda q, p: torch.cos(q)),
    ("sin(p)",    lambda q, p: torch.sin(p)),
    ("cos(p)",    lambda q, p: torch.cos(p)),
    ("q^3",       lambda q, p: q ** 3),
    ("p^3",       lambda q, p: p ** 3),
    ("q^2*p",     lambda q, p: q * q * p),
    ("q*p^2",     lambda q, p: q * p * p),
    ("exp(-q^2)", lambda q, p: torch.exp(-q * q)),
    ("|p|",       lambda q, p: p.abs()),
]


class BasisFeatures(nn.Module):
    """Apply every basis function in the library to (q, p) and stack."""

    def __init__(self, library=BASIS_LIBRARY):
        super().__init__()
        self.library = library
        self.names = [name for name, _ in library]

    def forward(self, qp: torch.Tensor) -> torch.Tensor:
        q = qp[..., 0:1]
        p = qp[..., 1:2]
        feats = [fn(q, p) for _, fn in self.library]
        return torch.cat(feats, dim=-1)


class _STESign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w):
        # Threshold at 0.5 * mean(|w|) — values below that snap to 0.
        eps = 1e-6
        thresh = 0.5 * w.abs().mean().clamp_min(eps)
        return torch.where(w.abs() < thresh, torch.zeros_like(w), torch.sign(w))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def ste_sign(w):
    return _STESign.apply(w)


class TernaryEQL(nn.Module):
    """Ternary Equation Learner.

    H(q, p) = sum_k  s_k * c_k * basis_k(q, p)  + bias
    where s_k in {-1, 0, +1} is a ternary selector (per-feature sign/skip)
    and c_k is a per-feature continuous coefficient.

    Why per-feature scale? With a single shared alpha (as in a plain
    BitLinear) we couldn't express two basis functions with different
    coefficient magnitudes simultaneously (e.g., 0.5*p^2 AND 1.0*cos(q)).
    A per-feature scale fixes that without losing the discrete selection:
    the ternary mask still says "include or skip", and each included term
    has a single human-readable real coefficient.
    """

    def __init__(self, library=BASIS_LIBRARY, ternary: bool = True):
        super().__init__()
        self.basis = BasisFeatures(library)
        n_basis = len(library)
        self.ternary = ternary
        if ternary:
            # Continuous magnitudes c_k (one per basis) + ternary selectors s_k
            # with STE backward.
            self.coef = nn.Parameter(torch.randn(n_basis) * 0.1)
            self.selector = nn.Parameter(torch.randn(n_basis) * 0.1)
            self.bias = nn.Parameter(torch.zeros(1))
        else:
            self.combine = nn.Linear(n_basis, 1)

    def forward(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        qp = torch.cat([q, p], dim=-1)
        feats = self.basis(qp)  # (..., n_basis)
        if self.ternary:
            s = ste_sign(self.selector)  # (n_basis,) in {-1, 0, +1}
            effective = self.coef * s    # (n_basis,) — sparse signed coefs
            return (feats * effective).sum(dim=-1) + self.bias

        return self.combine(feats).squeeze(-1)

    @torch.no_grad()
    def readable_formula(self) -> str:
        terms = []
        if self.ternary:
            s = ste_sign(self.selector)
            effective = (self.coef * s).tolist()
            bias = float(self.bias.item())
            if abs(bias) > 1e-3:
                terms.append(f"{bias:+.4f}")
            for c, name in zip(effective, self.basis.names):
                if c == 0:
                    continue
                terms.append(f"{c:+.4f} * {name}")
        else:
            row = self.combine.weight.squeeze(0).tolist()
            bias = float(self.combine.bias.item())
            if abs(bias) > 1e-3:
                terms.append(f"{bias:+.4f}")
            for w, name in zip(row, self.basis.names):
                if abs(w) < 1e-3:
                    continue
                terms.append(f"{w:+.4f} * {name}")
        return " ".join(terms) if terms else "0"

    @torch.no_grad()
    def active_basis_count(self) -> int:
        """How many basis functions did the discovered formula keep?"""
        if self.ternary:
            return int((ste_sign(self.selector) != 0).sum().item())
        return int((self.combine.weight.abs() > 1e-3).sum().item())

    def vector_field(self, q: torch.Tensor, p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = q.clone().detach().requires_grad_(True)
        p = p.clone().detach().requires_grad_(True)
        H = self.forward(q, p).sum()
        dHdq, dHdp = torch.autograd.grad(H, (q, p), create_graph=True)
        return dHdp, -dHdq
