"""Confidence-routed anytime inference for delta-sigma networks.

The idea: at each query, run the model at progressively larger k. Use
an output-confidence signal to decide whether to stop early or continue.
The router exposes a single threshold τ; per-query k is selected
automatically.

Confidence signals supported:
  - "entropy" : softmax entropy at the last token (smaller = more confident)
  - "topk_gap": probability margin between top-1 and top-2 logits
  - "diff"    : sup-norm change between successive k truncations (no quality
                supervision needed)

This is the missing piece that turns the runtime precision knob into
adaptive compute. With a good confidence signal, easy queries converge
at low k, hard queries spend full T.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from .dsigma_linear import DeltaSigmaLinear
from .dsigma_transformer import DSigmaCharLM


def _gather_streams(model: DSigmaCharLM) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Precompute all DeltaSigmaLinear streams once."""
    out = []
    for mod in model.modules():
        if isinstance(mod, DeltaSigmaLinear):
            out.append(mod.get_stream())  # (stream, alpha)
    return out


def _set_layer_truncation(model: DSigmaCharLM, k: int) -> None:
    """Temporarily monkey-patch each DSigmaLinear's effective T."""
    for mod in model.modules():
        if isinstance(mod, DeltaSigmaLinear):
            mod._truncation_k = k


def _restore_truncation(model: DSigmaCharLM) -> None:
    for mod in model.modules():
        if isinstance(mod, DeltaSigmaLinear):
            if hasattr(mod, "_truncation_k"):
                del mod._truncation_k


@torch.no_grad()
def confidence_router(
    model: DSigmaCharLM,
    idx: torch.Tensor,
    k_schedule: list[int] = (1, 2, 4, 8, 16, 32),
    signal: str = "entropy",
    threshold: float = 0.5,
) -> tuple[torch.Tensor, int]:
    """Run inference with progressively larger k until confident.

    Returns (logits_at_chosen_k, k_used).

    Signals:
      entropy : softmax entropy < threshold  (smaller = more confident)
      topk_gap: top1 - top2 > threshold       (larger gap = more confident)
      diff    : ||logits_k - logits_{prev}||_inf < threshold

    The "diff" signal uses no external supervision — it's the same
    early-exit criterion as in anytime_inference but applied to the
    transformer's output logits.
    """
    streams_and_alpha = _gather_streams(model)
    # We replace the layers' forward with a closure that uses the chosen k.
    # Simpler: rebuild streams per call and override mean computation.

    def forward_with_k(k: int):
        # Hot-patch every DeltaSigmaLinear to use only first k slices.
        for mod, (stream, alpha) in zip(
            [m for m in model.modules() if isinstance(m, DeltaSigmaLinear)],
            streams_and_alpha,
        ):
            mod._cached_stream = stream
            mod._cached_alpha = alpha
            mod._truncation_k = k

        # Re-implement the forward locally:
        from torch.nn import functional as F2
        # We exploit that DeltaSigmaLinear.forward() runs encode each call.
        # Instead, do a custom forward over the model that uses the truncated
        # cumulative mean. Easiest is to override DeltaSigmaLinear.forward
        # via monkey-patching for the duration.
        return model(idx)

    # The previous forward_with_k is correct only if DeltaSigmaLinear.forward
    # is set up to honor _truncation_k. We monkey-patch it temporarily here.
    original_forward = DeltaSigmaLinear.forward

    def patched_forward(self, x):
        x = self.norm(x)
        if hasattr(self, "_cached_stream"):
            stream = self._cached_stream
            alpha = self._cached_alpha
        else:
            stream, alpha = self.get_stream()
            self._cached_stream = stream
            self._cached_alpha = alpha
        k = getattr(self, "_truncation_k", self.T)
        w_eff = stream[:k].mean(dim=0) * alpha
        return F.linear(x, w_eff, self.bias)

    DeltaSigmaLinear.forward = patched_forward

    try:
        prev_logits = None
        chosen_k = k_schedule[-1]
        for k in k_schedule:
            logits = forward_with_k(k)
            confident = False
            if signal == "entropy":
                probs = F.softmax(logits[:, -1, :], dim=-1)
                ent = -(probs * (probs.clamp_min(1e-12)).log()).sum(dim=-1).mean().item()
                if ent < threshold:
                    confident = True
            elif signal == "topk_gap":
                probs = F.softmax(logits[:, -1, :], dim=-1)
                top2 = probs.topk(2, dim=-1).values
                gap = (top2[:, 0] - top2[:, 1]).mean().item()
                if gap > threshold:
                    confident = True
            elif signal == "diff":
                if prev_logits is not None:
                    delta = (logits - prev_logits).abs().max().item()
                    if delta < threshold:
                        confident = True
                prev_logits = logits
            else:
                raise ValueError(signal)
            if confident:
                chosen_k = k
                break
            chosen_k = k
        return logits, chosen_k
    finally:
        # Clean up cached state
        for mod in model.modules():
            if isinstance(mod, DeltaSigmaLinear):
                for attr in ("_cached_stream", "_cached_alpha", "_truncation_k"):
                    if hasattr(mod, attr):
                        delattr(mod, attr)
        DeltaSigmaLinear.forward = original_forward
