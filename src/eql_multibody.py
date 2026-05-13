"""Equation Learner with a multi-body basis library.

For 3-mass dynamics we need a larger basis: per-particle quadratics and
cross-particle products. We list ~30 candidate terms and let the ternary
selector pick the right subset.

True H expanded:
    H = 0.5 (p1^2 + p2^2 + p3^2) + (q1^2 + q2^2 + q3^2) - q1*q2 - q2*q3

So the correct sparse selection is *exactly 8 of these basis terms*.
A ternary EQL with per-feature scale should rediscover that subset.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .eql import ste_sign


def build_chain_basis_library():
    names = []
    fns = []

    # Per-particle quadratic momentum terms
    for i in range(3):
        names.append(f"p{i+1}^2")
        fns.append((lambda i: lambda q, p: p[..., i:i+1] ** 2)(i))

    # Per-particle quadratic position terms
    for i in range(3):
        names.append(f"q{i+1}^2")
        fns.append((lambda i: lambda q, p: q[..., i:i+1] ** 2)(i))

    # Cross-particle position products (pairs)
    for i in range(3):
        for j in range(i + 1, 3):
            names.append(f"q{i+1}*q{j+1}")
            fns.append((lambda i, j: lambda q, p: q[..., i:i+1] * q[..., j:j+1])(i, j))

    # Cross-particle momentum products
    for i in range(3):
        for j in range(i + 1, 3):
            names.append(f"p{i+1}*p{j+1}")
            fns.append((lambda i, j: lambda q, p: p[..., i:i+1] * p[..., j:j+1])(i, j))

    # q*p mixed terms
    for i in range(3):
        names.append(f"q{i+1}*p{i+1}")
        fns.append((lambda i: lambda q, p: q[..., i:i+1] * p[..., i:i+1])(i))

    # Linear per-particle terms (shouldn't be needed but provide noise)
    for i in range(3):
        names.append(f"q{i+1}")
        fns.append((lambda i: lambda q, p: q[..., i:i+1])(i))

    # Quartic in q (shouldn't be needed)
    for i in range(3):
        names.append(f"q{i+1}^4")
        fns.append((lambda i: lambda q, p: q[..., i:i+1] ** 4)(i))

    return names, fns


class ChainBasisFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.names, self.fns = build_chain_basis_library()

    def forward(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return torch.cat([fn(q, p) for fn in self.fns], dim=-1)


class ChainEQL(nn.Module):
    """Ternary EQL over the multi-body basis library.

    H(q, p) = sum_k  s_k * c_k * basis_k(q, p)
    where s_k in {-1, 0, +1} (ternary selector) and c_k is a continuous
    per-feature magnitude. Identical structure to the pendulum EQL but
    over a 24-term library.
    """

    def __init__(self, ternary: bool = True):
        super().__init__()
        self.basis = ChainBasisFeatures()
        n = len(self.basis.names)
        self.ternary = ternary
        if ternary:
            self.coef = nn.Parameter(torch.randn(n) * 0.1)
            self.selector = nn.Parameter(torch.randn(n) * 0.1)
            self.bias = nn.Parameter(torch.zeros(1))
        else:
            self.combine = nn.Linear(n, 1)

    def forward(self, q, p):
        feats = self.basis(q, p)
        if self.ternary:
            s = ste_sign(self.selector)
            effective = self.coef * s
            return (feats * effective).sum(dim=-1) + self.bias
        return self.combine(feats).squeeze(-1)

    def vector_field(self, q, p):
        q = q.clone().detach().requires_grad_(True)
        p = p.clone().detach().requires_grad_(True)
        H = self.forward(q, p).sum()
        dHdq, dHdp = torch.autograd.grad(H, (q, p), create_graph=True)
        return dHdp, -dHdq

    @torch.no_grad()
    def readable_formula(self):
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
                if abs(w) < 5e-3:
                    continue
                terms.append(f"{w:+.4f} * {name}")
        return " ".join(terms) if terms else "0"

    @torch.no_grad()
    def active_basis_count(self):
        if self.ternary:
            return int((ste_sign(self.selector) != 0).sum().item())
        return int((self.combine.weight.abs() > 5e-3).sum().item())
