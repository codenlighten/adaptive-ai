"""Pretrain a fp32 LM, then continue training as a ternary model.

Compare:
  A. BitLM trained from random init (baseline — what we already have)
  B. FPLM trained from random init, then ternarized + continued training

In real BitNet workflows, (B) is the standard recipe: pretrain in higher
precision, then do quantization-aware fine-tuning. We test whether the
fp-pretrained initialization gives the ternary model a head start.

Run: venv/bin/python -m scripts.finetune_lm
"""

from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.char_lm import BitCharLM, FPCharLM
from src.finetune import copy_fp_to_bit_char_lm
from src.physics_corpus import CharVocab, build_corpus, make_batches


def train_one(model, batch_iter, val_x, val_y, steps, lr, label):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    history = []
    t0 = time.time()
    for step in range(steps):
        model.train()
        x, y = next(batch_iter)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % max(1, steps // 12) == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                vl = F.cross_entropy(
                    model(val_x).reshape(-1, model.head.out_features),
                    val_y.reshape(-1),
                ).item()
            history.append((step + 1, vl))
            print(f"[{label}] step {step+1:5d}/{steps}  "
                  f"train={loss.item():.3f}  val={vl:.3f}  ppl={math.exp(vl):.2f}")
    print(f"[{label}] done in {time.time()-t0:.1f}s")
    return history


def eval_model(model, x, y):
    model.eval()
    with torch.no_grad():
        loss = F.cross_entropy(
            model(x).reshape(-1, model.head.out_features),
            y.reshape(-1),
        ).item()
    return math.exp(loss)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain-steps", type=int, default=600)
    parser.add_argument("--finetune-steps", type=int, default=600)
    parser.add_argument("--scratch-steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--ft-lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-lines", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    text = build_corpus(args.n_lines, seed=args.seed)
    print(f"Corpus: {len(text):,} chars")
    vocab = CharVocab(text)
    print(f"Vocab: {vocab.size}\n")

    split = int(len(text) * 0.9)
    train_text, val_text = text[:split], text[split:]
    train_iter = lambda seed: make_batches(train_text, vocab, args.block_size,
                                            args.batch_size, seed=seed)
    val_iter = make_batches(val_text, vocab, args.block_size, args.batch_size * 2,
                            seed=args.seed + 100)
    val_x, val_y = next(val_iter)

    # ---- Phase 1: fp32 pretrain ----
    print("=" * 70)
    print("Phase A: pretrain fp32 LM")
    print("=" * 70)
    fp = FPCharLM(vocab.size, d_model=args.d_model, n_heads=args.n_heads,
                  n_layers=args.n_layers, max_len=args.block_size)
    train_one(fp, train_iter(args.seed), val_x, val_y, args.pretrain_steps,
              args.lr, "PreFP")
    pretrain_fp_ppl = eval_model(fp, val_x, val_y)
    print(f"\n  Pretrained FP val ppl: {pretrain_fp_ppl:.3f}\n")

    # ---- Phase 2: ternarize and continue ----
    print("=" * 70)
    print("Phase B: copy weights to BitLM and continue training (QAT-style)")
    print("=" * 70)
    bit_from_fp = BitCharLM(vocab.size, d_model=args.d_model, n_heads=args.n_heads,
                            n_layers=args.n_layers, max_len=args.block_size)
    copy_fp_to_bit_char_lm(fp, bit_from_fp)
    init_ternary_ppl = eval_model(bit_from_fp, val_x, val_y)
    print(f"  Ternary ppl right after weight copy (before any fine-tune): "
          f"{init_ternary_ppl:.3f}")
    train_one(bit_from_fp, train_iter(args.seed + 1), val_x, val_y,
              args.finetune_steps, args.ft_lr, "BitFT")
    ft_ppl = eval_model(bit_from_fp, val_x, val_y)
    print(f"\n  Fine-tuned ternary ppl: {ft_ppl:.3f}\n")

    # ---- Phase 3: ternary-from-scratch baseline ----
    print("=" * 70)
    print("Phase C: BitLM trained from scratch (baseline)")
    print("=" * 70)
    torch.manual_seed(args.seed + 42)
    bit_scratch = BitCharLM(vocab.size, d_model=args.d_model, n_heads=args.n_heads,
                            n_layers=args.n_layers, max_len=args.block_size)
    train_one(bit_scratch, train_iter(args.seed + 2), val_x, val_y,
              args.scratch_steps, args.lr, "BitSC")
    scratch_ppl = eval_model(bit_scratch, val_x, val_y)
    print(f"\n  From-scratch ternary ppl: {scratch_ppl:.3f}\n")

    # ---- Comparison ----
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  FP-pretrained ({args.pretrain_steps} steps): ppl = {pretrain_fp_ppl:.3f}")
    print(f"  After ternarization (no fine-tune):           ppl = {init_ternary_ppl:.3f}")
    print(f"  After {args.finetune_steps}-step ternary fine-tune:        ppl = {ft_ppl:.3f}")
    print(f"  Ternary from scratch ({args.scratch_steps} steps):           ppl = {scratch_ppl:.3f}")
    print(f"\n  pretrain+ft total steps: {args.pretrain_steps + args.finetune_steps}")
    print(f"  scratch    total steps: {args.scratch_steps}")
    if ft_ppl < scratch_ppl:
        print(f"\n  -> Pretrain+ternarize wins by {(scratch_ppl - ft_ppl):.3f} ppl.")
    else:
        print(f"\n  -> From-scratch wins (or ties) at this scale.")


if __name__ == "__main__":
    main()
