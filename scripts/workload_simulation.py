"""Realistic workload simulation: heterogeneous queries through a
confidence-routed DSigma transformer, projected to hyperscale costs.

We construct three "query difficulty buckets" by taking validation
sequences and stratifying them by next-token entropy under a fully-
trained model. Then we route them through DSigma with varying
confidence thresholds and measure:

  - the per-query k distribution
  - the average compute reduction
  - the quality (perplexity) on each bucket

Finally we extrapolate to hyperscale electricity costs.

Run: venv/bin/python -m scripts.workload_simulation
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from delta_sigma_nn.dsigma_router import confidence_router
from delta_sigma_nn.dsigma_transformer import DSigmaCharLM
from src.physics_corpus import CharVocab, build_corpus, make_batches


def stratify_by_difficulty(model, val_data, vocab, n_buckets=3, n_per_bucket=80,
                            block_size=64):
    """Score every validation window by next-token entropy under the model.
    Return n_per_bucket samples per (easy, medium, hard) bucket."""
    model.eval()
    entropies = []
    contexts = []
    with torch.no_grad():
        for i in range(0, val_data.shape[0] - block_size - 1, 4):
            x = val_data[i:i+block_size].unsqueeze(0)
            y_next = val_data[i+block_size:i+block_size+1]
            logits = model(x)[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            ent = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1).item()
            entropies.append(ent)
            contexts.append((x, y_next))

    # Sort by entropy
    paired = sorted(zip(entropies, contexts), key=lambda p: p[0])
    n = len(paired)
    print(f"  scored {n} validation windows  (entropy range: "
          f"{paired[0][0]:.3f} - {paired[-1][0]:.3f})")
    third = n // 3
    easy   = paired[:third][:n_per_bucket]
    medium = paired[third:2*third][:n_per_bucket]
    hard   = paired[2*third:][-n_per_bucket:]
    return easy, medium, hard


def run_router_sweep(model, samples, thresholds, signal="diff", k_schedule=(2,4,8)):
    """Run inference on each sample at each threshold; return (k_used, correct) per sample."""
    results = []
    for thr in thresholds:
        k_list, correct_list, nll_list = [], [], []
        for ent, (x, y_next) in samples:
            logits, k = confidence_router(model, x, k_schedule=list(k_schedule),
                                           signal=signal, threshold=thr)
            pred = logits[0, -1, :].argmax().item()
            correct = int(pred == y_next.item())
            nll = F.cross_entropy(logits[0, -1:], y_next).item()
            k_list.append(k)
            correct_list.append(correct)
            nll_list.append(nll)
        results.append({
            "threshold": thr,
            "avg_k": sum(k_list) / len(k_list),
            "max_k": max(k_list),
            "p99_k": sorted(k_list)[int(0.99 * len(k_list))],
            "accuracy": sum(correct_list) / len(correct_list),
            "avg_nll": sum(nll_list) / len(nll_list),
        })
    return results


def main():
    torch.manual_seed(0)
    text = build_corpus(6000, seed=0)
    vocab = CharVocab(text)
    split = int(len(text) * 0.9)
    val_text = text[split:]
    val_data = torch.tensor(vocab.encode(val_text), dtype=torch.long)
    print(f"Validation data: {val_data.shape[0]} tokens")

    # Use the same architecture as our 556k transformer experiment
    model = DSigmaCharLM(vocab.size, d_model=128, n_heads=4, n_layers=4,
                         max_len=64, T=8)

    print("Training model (500 steps)...")
    train_data = torch.tensor(vocab.encode(text[:split]), dtype=torch.long)
    train_iter = make_batches(text[:split], vocab, 64, 64, seed=0)

    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=500)
    t0 = time.time()
    for step in range(500):
        x, y = next(train_iter)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    print(f"  trained in {time.time()-t0:.0f}s")

    print("\nStratifying validation windows into easy/medium/hard buckets...")
    easy, medium, hard = stratify_by_difficulty(model, val_data, vocab,
                                                 n_per_bucket=20, block_size=64)
    print(f"  easy:   {len(easy)} samples  (top tertile of confidence)")
    print(f"  medium: {len(medium)} samples")
    print(f"  hard:   {len(hard)} samples  (bottom tertile)")

    thresholds = [5.0, 2.0, 0.5, 0.1, 0.02]
    print("\n" + "="*78)
    print("Confidence-routed inference (signal='diff'):")
    print("="*78)
    for name, samples in [("EASY  ", easy), ("MEDIUM", medium), ("HARD  ", hard)]:
        print(f"\n[{name}] {len(samples)} samples:")
        rows = run_router_sweep(model, samples, thresholds)
        print(f"  {'threshold':>10}  {'avg_k':>6}  {'max_k':>6}  "
              f"{'acc':>6}  {'avg_nll':>8}")
        for r in rows:
            print(f"  {r['threshold']:>10.3f}  {r['avg_k']:>6.2f}  {r['max_k']:>6d}  "
                  f"{r['accuracy']:>6.3f}  {r['avg_nll']:>8.4f}")

    # ----- Combined workload model -----
    print("\n" + "="*78)
    print("Combined workload (70% easy / 25% medium / 5% hard):")
    print("="*78)
    workload = easy[:14] + medium[:5] + hard[:1]  # 70/25/5 of 20
    rows = run_router_sweep(model, workload, thresholds)
    print(f"  {'threshold':>10}  {'avg_k':>6}  {'speedup':>8}  {'acc':>6}  {'avg_nll':>8}")
    base_nll = rows[-1]["avg_nll"]
    for r in rows:
        speedup = 8 / r["avg_k"]
        print(f"  {r['threshold']:>10.3f}  {r['avg_k']:>6.2f}  {speedup:>7.2f}x  "
              f"{r['accuracy']:>6.3f}  {r['avg_nll']:>8.4f}")

    # ----- Hyperscale projection -----
    print("\n" + "="*78)
    print("Hyperscale energy projection")
    print("="*78)
    # Pick the threshold that loses <2% accuracy
    base_acc = rows[-1]["accuracy"]
    chosen = None
    for r in rows:
        if r["accuracy"] >= base_acc - 0.02:
            chosen = r
    print(f"  Operating point: threshold = {chosen['threshold']:.3f}")
    print(f"    avg_k = {chosen['avg_k']:.2f} of 8  (compute reduction "
          f"{8/chosen['avg_k']:.2f}x)")
    print(f"    accuracy {chosen['accuracy']:.3f} vs full-k baseline {base_acc:.3f}")

    # Industry numbers (rough, early 2026)
    print("\n  Rough industry numbers (early 2026):")
    print("    Global AI inference electricity:   ~1 GW  (8.8 TWh/yr)")
    print("    Electricity rate (industrial):     ~$0.08 / kWh")
    print("    Total annual inference $:          ~$700 M (compute only)")
    print("    With PUE 1.4 for cooling:          ~$1.0 B")
    reduction = 8 / chosen['avg_k']
    print(f"\n  At {reduction:.2f}x compute reduction (matching this run):")
    print(f"    Inference $: ~${1_000_000_000 * (1 - 1/reduction)/1e6:,.0f} M / year savings industry-wide")
    print(f"    or per-hyperscaler @ 15% market share: "
          f"~${1_000_000_000 * 0.15 * (1 - 1/reduction)/1e6:,.0f} M / year")
    print()
    print("  Caveats:")
    print("  - These numbers assume industry-wide adoption and a workload mix")
    print("    that matches our 70/25/5 stratification. Real workloads vary.")
    print("  - Validation here is at 556k params; LLM-scale validation pending.")
    print("  - Quality eval is next-token-accuracy on synthetic physics text,")
    print("    not a production benchmark like MMLU/HellaSwag.")


if __name__ == "__main__":
    main()
