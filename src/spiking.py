"""Spiking ternary neurons: LIF integrate-and-fire with {-1, 0, +1} spikes.

Each "neuron" has a membrane potential V that integrates input current
over T timesteps. At each step it emits a spike:
    +1 if V crosses +theta
    -1 if V crosses -theta (signed/balanced LIF, the natural ternary extension)
     0 otherwise

After spiking, V resets toward 0 by subtracting the spike. The whole
network's "output" at each timestep is a vector of trits; the final
prediction is the time-averaged output over T steps.

This is closer to neuromorphic hardware than the stochastic-trit version
(phase 12): the trits emerge from real temporal dynamics, not random
sampling. And it's still ternary {-1, 0, +1}.

Surrogate gradient: in backward we replace the non-differentiable
threshold step with the derivative of a smoothed function (we use the
gradient of Hardtanh, clipped to |V| <= theta).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ternary import BitLinear


class _SignSpike(torch.autograd.Function):
    """Forward: sign(V) where |V| > theta else 0. Backward: clipped pass-through."""

    @staticmethod
    def forward(ctx, V, theta):
        ctx.save_for_backward(V)
        ctx.theta = float(theta)
        return torch.where(V.abs() < theta, torch.zeros_like(V), torch.sign(V))

    @staticmethod
    def backward(ctx, grad_output):
        (V,) = ctx.saved_tensors
        # Surrogate: derivative is 1 inside (-1.5*theta, 1.5*theta), else 0.
        mask = (V.abs() <= 1.5 * ctx.theta).to(grad_output.dtype)
        return grad_output * mask, None


def sign_spike(V, theta):
    return _SignSpike.apply(V, theta)


class SpikingTernaryLayer(nn.Module):
    """One LIF layer with BitLinear weight matrix and signed spike output."""

    def __init__(self, in_features, out_features, decay=0.9, theta=0.5):
        super().__init__()
        self.linear = BitLinear(in_features, out_features)
        self.decay = decay
        self.theta = theta

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """x_seq shape: (T, B, in_features). Returns (T, B, out_features) of trits."""
        T, B, _ = x_seq.shape
        V = torch.zeros(B, self.linear.out_features, device=x_seq.device,
                        dtype=x_seq.dtype)
        spikes = []
        for t in range(T):
            current = self.linear(x_seq[t])
            V = self.decay * V + current
            s = sign_spike(V, self.theta)
            V = V - s * self.theta * 2.0   # soft reset
            spikes.append(s)
        return torch.stack(spikes, dim=0)


class SpikingTernaryMLP(nn.Module):
    """Encode a static input as T identical copies, run through L spiking layers,
    return the time-averaged final layer output."""

    def __init__(self, in_dim, hidden, out_dim, depth=4, T=8, decay=0.9, theta=0.5):
        super().__init__()
        self.T = T
        self.fc_in = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList([
            SpikingTernaryLayer(hidden, hidden, decay=decay, theta=theta)
            for _ in range(depth - 2)
        ])
        self.fc_out = nn.Linear(hidden, out_dim)

    def forward(self, x):
        h = F.gelu(self.fc_in(x))
        # Replicate input across T timesteps
        h_seq = h.unsqueeze(0).expand(self.T, *h.shape).contiguous()
        for layer in self.layers:
            h_seq = layer(h_seq)
        # Time-average the trit stream, then project out
        h_avg = h_seq.mean(dim=0)
        return self.fc_out(h_avg)
