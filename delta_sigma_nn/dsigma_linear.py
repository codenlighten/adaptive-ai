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
Gradient is not routed through `alpha = |W|.mean()` — this is the standard
BitNet b1.58 approximation and works well in practice.

The key claim: at sufficiently large T, the effective weight is arbitrarily
close to the underlying float value, but each of the T matmuls used zero
floating-point multiplications (in a multiply-free backend; see
`dsigma_pack.dsigma_inference` for the reference NumPy oracle which uses
a single fused matmul for simplicity). The total compute is T one-trit
matmuls plus one scalar multiply by alpha at the end.

There's an additional inference trick this enables: anytime inference.
The cumulative average over the first k of T steps is a progressively
better estimate. You can stop early when the output stops changing. See
`dsigma_inference_context` below.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from .delta_sigma import encode_delta_sigma_order2, encode_delta_sigma_ternary


class _STEEncode(torch.autograd.Function):
    """STE around the delta-sigma encode + time-average.

    Forward computes (mean(stream) * alpha). Backward returns the upstream
    gradient straight through to W with no routing through alpha — matching
    standard BitNet b1.58 practice. Empirically stable; analytic alpha-grad
    can be added later if mixed-precision training surfaces issues.
    """

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

    Inference-time precision knob: when a DeltaSigmaLinear is wrapped in
    `dsigma_inference_context(model, k=K)`, each forward uses only the
    first K of T stream slices and skips re-encoding entirely (the stream
    is cached on the instance). All caching state is per-instance and
    cleared on context exit — no class-level mutation, safe across
    successive calls.
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
        cached_stream = getattr(self, "_cached_stream", None)
        if cached_stream is not None:
            # Inference path: stream is precomputed, k may be truncated.
            alpha = self._cached_alpha
            k = getattr(self, "_truncation_k", None)
            k_eff = self.T if k is None else min(k, self.T)
            w_eff = cached_stream[:k_eff].mean(dim=0) * alpha
            return F.linear(x, w_eff, self.bias)
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


@contextmanager
def dsigma_inference_context(model: nn.Module, k: int | None = None) -> Iterator[list["DeltaSigmaLinear"]]:
    """Cache delta-sigma streams on every DeltaSigmaLinear in `model`.

    Inside the `with` block, every DeltaSigmaLinear forward uses the cached
    stream truncated to `k` time steps (or full `T` if k is None). On exit,
    all cached state is removed from each module — the class is never
    mutated, so concurrent contexts on disjoint models are safe.

    Yields the list of touched DeltaSigmaLinear modules so callers can
    iterate them (e.g. to change `_truncation_k` mid-context).

    Example:
        with dsigma_inference_context(model, k=4):
            y = model(x)              # uses k=4 everywhere
        # outside the block, model() uses the training path again
    """
    ds_layers: list[DeltaSigmaLinear] = [
        m for m in model.modules() if isinstance(m, DeltaSigmaLinear)
    ]
    try:
        for m in ds_layers:
            stream, alpha = m.get_stream()
            m._cached_stream = stream
            m._cached_alpha = alpha
            if k is not None:
                m._truncation_k = k
        yield ds_layers
    finally:
        for m in ds_layers:
            for attr in ("_cached_stream", "_cached_alpha", "_truncation_k"):
                if hasattr(m, attr):
                    delattr(m, attr)


class DeltaSigmaMLP(nn.Module):
    """MLP with fp32 input/output boundary layers and delta-sigma hidden layers.

    Architecture:
        in_proj (fp32)  -> GELU -> [DeltaSigmaLinear -> GELU] x (depth-2) -> out_proj (fp32)

    Boundary layers are exposed as `in_proj` and `out_proj` so serialization
    and analysis code can address them by name; the hidden delta-sigma layers
    are in `dsigma_blocks` (an `nn.ModuleList`).
    """

    def __init__(self, in_dim, hidden_dim, out_dim, depth=5, T=8, order=1):
        super().__init__()
        self.T = T
        self.order = order
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.dsigma_blocks = nn.ModuleList([
            DeltaSigmaLinear(hidden_dim, hidden_dim, T=T, order=order)
            for _ in range(depth - 2)
        ])
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        h = F.gelu(self.in_proj(x))
        for blk in self.dsigma_blocks:
            h = F.gelu(blk(h))
        return self.out_proj(h)

    def dsigma_layers(self) -> list["DeltaSigmaLinear"]:
        return list(self.dsigma_blocks)

    @torch.no_grad()
    def anytime_inference(self, x: torch.Tensor, T_max: int | None = None,
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
        with dsigma_inference_context(self) as ds_layers:
            def forward_with_k(k: int) -> torch.Tensor:
                for m in ds_layers:
                    m._truncation_k = k
                return self(x)

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
