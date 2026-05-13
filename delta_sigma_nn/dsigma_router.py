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

Implementation note: this uses `dsigma_inference_context` to cache the
delta-sigma streams on each DeltaSigmaLinear *instance* once, then varies
`_truncation_k` per iteration. No class-level mutation — concurrent
routers on disjoint models are safe.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dsigma_linear import DeltaSigmaLinear, dsigma_inference_context


def _is_confident(logits: torch.Tensor, prev_logits: torch.Tensor | None,
                  signal: str, threshold: float) -> bool:
    if signal == "entropy":
        probs = F.softmax(logits[:, -1, :], dim=-1)
        ent = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1).mean().item()
        return ent < threshold
    if signal == "topk_gap":
        probs = F.softmax(logits[:, -1, :], dim=-1)
        top2 = probs.topk(2, dim=-1).values
        gap = (top2[:, 0] - top2[:, 1]).mean().item()
        return gap > threshold
    if signal == "diff":
        if prev_logits is None:
            return False
        return (logits - prev_logits).abs().max().item() < threshold
    raise ValueError(f"unknown signal {signal!r}")


@torch.no_grad()
def confidence_router(
    model: nn.Module,
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

    Works on any nn.Module containing DeltaSigmaLinear layers — not
    restricted to DSigmaCharLM. The router caches each layer's stream
    once and varies the truncation per iteration, so the cost is
    encode-once + len(k_schedule) cheap forwards.
    """
    with dsigma_inference_context(model) as ds_layers:
        prev_logits: torch.Tensor | None = None
        logits: torch.Tensor | None = None
        chosen_k = k_schedule[-1]
        for k in k_schedule:
            for m in ds_layers:
                m._truncation_k = k
            logits = model(idx)
            chosen_k = k
            if _is_confident(logits, prev_logits, signal, threshold):
                break
            prev_logits = logits
    assert logits is not None, "k_schedule must be non-empty"
    return logits, chosen_k
