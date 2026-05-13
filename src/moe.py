"""Ternary Mixture of Experts.

K small ternary expert MLPs + a learnable router. The router sends each
input to its top-k experts (k=1 or 2) and averages their outputs by the
router weights. Both experts and router use BitLinear hidden layers.

Why this is interesting for ternary:
  - Per-input sparsity (only k of K experts active) composes with
    per-weight sparsity (ternary {-1, 0, +1} has ~33-50% zeros).
  - Total trainable params scale with K, but FLOPs per forward stay flat.
  - The router's discrete decisions are conceptually aligned with the
    ternary weights' discrete values — both make hard choices.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ternary import BitLinear


class TernaryExpert(nn.Module):
    """A small ternary-hidden MLP."""

    def __init__(self, in_dim, hidden, out_dim, depth=3):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.GELU()]
        for _ in range(depth - 2):
            layers += [BitLinear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TernaryMoE(nn.Module):
    """Top-k routed mixture of K ternary experts.

    The router is a small fp linear layer (very few params) that outputs
    logits over K experts. We take top-k, softmax over those, and use
    them as weights for the per-expert outputs.
    """

    def __init__(self, in_dim, hidden, out_dim, n_experts=4, top_k=2, depth=3):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = nn.Linear(in_dim, n_experts)
        self.experts = nn.ModuleList([
            TernaryExpert(in_dim, hidden, out_dim, depth=depth)
            for _ in range(n_experts)
        ])

    def forward(self, x):
        # router_logits: (B, K)
        router_logits = self.router(x)
        topk_vals, topk_idx = router_logits.topk(self.top_k, dim=-1)
        topk_weights = F.softmax(topk_vals, dim=-1)            # (B, k)

        # Run every expert on every input (simple/dense form), then mask
        # by the per-input top-k selection. Real MoE implementations
        # gather per-expert subsets; here we just zero out non-selected
        # experts. This is fine for K small and correct numerically.
        expert_outs = torch.stack([e(x) for e in self.experts], dim=1)  # (B, K, out)

        # Build a (B, K) mask of zeros and softmax weights
        weights = torch.zeros_like(router_logits)              # (B, K)
        weights.scatter_(1, topk_idx, topk_weights)
        return (weights.unsqueeze(-1) * expert_outs).sum(dim=1)

    def routing_stats(self, x: torch.Tensor) -> dict[str, float]:
        """Diagnostic: how balanced is expert usage across a batch?"""
        with torch.no_grad():
            logits = self.router(x)
            _, idx = logits.topk(self.top_k, dim=-1)
            counts = torch.zeros(self.n_experts)
            for i in idx.flatten().tolist():
                counts[i] += 1
            total = counts.sum().item()
            return {
                f"expert_{i}_frac": counts[i].item() / total
                for i in range(self.n_experts)
            }
