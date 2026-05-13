"""Stochastic ternary activation — unbiased {-1, 0, +1} sampling.

Phase 1 found that deterministic ternary activations collapse the network
output to a staircase. The fix is to make the activation stochastic so
that *on expectation* it equals the pre-activation. Then any single
forward pass produces a noisy but unbiased trit-stream, and averaging
N forward passes recovers the float signal with error ~1/sqrt(N).

Sampling rule for pre-activation y (after rescaling to roughly [-1, 1]
via a learnable temperature):
    if y >  0:  sample +1 with probability p+, else 0
               where p+ = min(y, 1)
    if y <  0:  sample -1 with probability p-, else 0
               where p- = min(-y, 1)
    if y == 0:  always 0

Expectation:  E[output] = +1 * p+ + 0 * (1-p+ -p-) + -1 * p-
                       = sign(y) * min(|y|, 1)
              = hardtanh(y),  which is locally linear in [-1, 1].

In the saturation regime |y| >= 1, the unbiased sample is just sign(y).
In the linear regime, ensemble averaging is a Monte Carlo estimator of
hardtanh(y).

Training uses STE (identity gradient on the sampling step). At inference
time, sampling N copies and averaging gives an analog-valued output that
*should* approach the float reference, removing the staircase.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _STEStochasticTrit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        # Clamp to [-1, 1] so probabilities make sense.
        p = x.clamp(-1.0, 1.0)
        # Draw uniform u in [0, 1) for each element.
        u = torch.rand_like(p)
        # Sample {-1, 0, +1}:
        #   p > 0: output +1 with prob p, else 0   (u < p means "fire +1")
        #   p < 0: output -1 with prob |p|, else 0 (u < |p| means "fire -1")
        out = torch.where(
            p > 0,
            torch.where(u < p, torch.ones_like(p), torch.zeros_like(p)),
            torch.where(u < -p, -torch.ones_like(p), torch.zeros_like(p)),
        )
        return out

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def stochastic_trit(x: torch.Tensor) -> torch.Tensor:
    return _STEStochasticTrit.apply(x)


class StochasticTernaryActivation(nn.Module):
    """Per-tensor adaptive temperature + stochastic {-1, 0, +1} sampling.

    The temperature is tau = mean(|x|) — keeps the pre-activation roughly
    in [-1, 1] on average so the saturated regime doesn't dominate.
    """

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            tau = x.detach().abs().mean().clamp_min(self.eps)
            return stochastic_trit(x / tau)
        # Inference: deterministic hardtanh approximation (E[stochastic_trit]
        # over the noise) — avoids variance unless caller wants ensemble samples.
        tau = x.detach().abs().mean().clamp_min(self.eps)
        return x.div(tau).clamp(-1.0, 1.0)

    def sample(self, x: torch.Tensor) -> torch.Tensor:
        """Force a stochastic sample even at eval time (for ensembling)."""
        tau = x.detach().abs().mean().clamp_min(self.eps)
        return stochastic_trit(x / tau)
