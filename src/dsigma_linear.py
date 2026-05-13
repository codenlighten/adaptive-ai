"""DeltaSigmaLinear: a linear layer whose weights are delta-sigma encoded.

At forward time:
  1. Normalize the weight matrix by its absolute mean (so it sits in [-1, 1]
     for the modulator) — call this scale alpha.
  2. Encode to a (T, out, in) tensor of trits via first-order delta-sigma.
  3. The "effective" weight is the mean of the stream times alpha. Equivalent
     to a T-step time-average of T one-trit matmuls.

For training we use a straight-through estimator on the encode step: the
forward uses the time-averaged trit reconstruction, the backward uses the
identity gradient with respect to the underlying float weight tensor.

The key claim: at sufficiently large T, the effective weight is arbitrarily
close to the underlying float value, but each of the T matmuls used zero
floating-point multiplications. The total compute is T one-trit matmuls
plus one scalar multiply by alpha at the end.

There's an additional inference trick this enables: anytime inference.
The cumulative average over the first k of T steps is a progressively
better estimate. You can stop early when the output stops changing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .delta_sigma import encode_delta_sigma_order2, encode_delta_sigma_ternary


class _STEEncode(torch.autograd.Function):
    """STE around the delta-sigma encode + time-average."""

    @staticmethod
    def forward(ctx, W: torch.Tensor, T: int, order: int) -> torch.Tensor:
        alpha = W.abs().mean().clamp_min(1e-5)
        W_norm = (W / alpha).clamp(-1.0, 1.0)
        if order == 1:
            stream = encode_delta_sigma_ternary(W_norm, T=T)
        else:
            stream = encode_delta_sigma_order2(W_norm, T=T)
        avg = stream.mean(dim=0)
        return avg * alpha

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def dsigma_encode(W, T, order=1):
    return _STEEncode.apply(W, T, order)


class DeltaSigmaLinear(nn.Module):
    """Linear layer with delta-sigma-encoded weights.

    Forward: y = LayerNorm(x) @ effective(W).T  + bias
    where effective(W) is the T-step delta-sigma time-average of W.

    At inference with T fixed, you can pre-compute the stream once and
    treat it as T separate ternary weight matrices. The forward becomes:
        y = (alpha / T) * sum_t (stream[t] @ x_norm.T)
    where each inner matmul uses zero multiplications.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 T: int = 8, order: int = 1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.T = T
        self.order = order
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.norm = nn.LayerNorm(in_features)
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x):
        x = self.norm(x)
        w = dsigma_encode(self.weight, self.T, self.order)
        return F.linear(x, w, self.bias)

    @torch.no_grad()
    def get_stream(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (stream, alpha) — useful for anytime inference."""
        alpha = self.weight.abs().mean().clamp_min(1e-5)
        W_norm = (self.weight / alpha).clamp(-1.0, 1.0)
        if self.order == 1:
            stream = encode_delta_sigma_ternary(W_norm, T=self.T)
        else:
            stream = encode_delta_sigma_order2(W_norm, T=self.T)
        return stream, alpha


class DeltaSigmaMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, depth=5, T=8, order=1):
        super().__init__()
        self.T = T
        self.order = order
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(depth - 2):
            layers += [DeltaSigmaLinear(hidden_dim, hidden_dim, T=T, order=order),
                       nn.GELU()]
        layers += [nn.Linear(hidden_dim, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def dsigma_layers(self):
        return [m for m in self.net if isinstance(m, DeltaSigmaLinear)]

    @torch.no_grad()
    def anytime_inference(self, x: torch.Tensor, T_max: int = None,
                          stop_eps: float = 1e-3) -> tuple[torch.Tensor, int]:
        """Run inference with progressively more time steps until output stabilizes.

        For each DeltaSigmaLinear layer we compute its full T-step stream
        once, then for k = 1, 2, 4, 8, ... up to T_max we replay the network
        using only the first k steps. We stop when the output change between
        successive k values drops below stop_eps.

        Returns (final_output, k_used).
        """
        T_max = T_max or self.T
        was_training = self.training
        self.eval()
        # Build the streams once per dsigma layer
        streams = []
        for m in self.dsigma_layers():
            streams.append(m.get_stream())  # (stream, alpha)

        def forward_with_k(k):
            h = x
            d_idx = 0
            for m in self.net:
                if isinstance(m, DeltaSigmaLinear):
                    stream, alpha = streams[d_idx]
                    d_idx += 1
                    h_norm = m.norm(h)
                    # average first k slices
                    w_eff = stream[:k].mean(dim=0) * alpha
                    h = F.linear(h_norm, w_eff, m.bias)
                else:
                    h = m(h)
            return h

        prev = forward_with_k(1)
        k = 1
        for k in [2, 4, 8, 16, 32, 64, 128]:
            if k > T_max:
                break
            cur = forward_with_k(k)
            delta = (cur - prev).abs().max().item()
            prev = cur
            if delta < stop_eps:
                break
        self.train(was_training)
        return prev, k
