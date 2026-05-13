"""Train ternary vs fp32 char-level LM on synthetic physics text.

Run: venv/bin/python -m src.train_char_lm
"""

from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .char_lm import BitCharLM, FPCharLM
from .physics_corpus import CharVocab, build_corpus, make_batches


def train(model, batch_iter, val_x, val_y, steps, lr, label):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    history = {"train": [], "val": []}
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

        if step % max(1, steps // 20) == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                vl = F.cross_entropy(
                    model(val_x).reshape(-1, model.head.out_features),
                    val_y.reshape(-1),
                ).item()
            history["train"].append(loss.item())
            history["val"].append(vl)
            print(f"[{label}] step {step+1:5d}/{steps}  "
                  f"train={loss.item():.3f}  val={vl:.3f}  ppl={math.exp(vl):.2f}")
    print(f"[{label}] done in {time.time()-t0:.1f}s")
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-lines", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    print("Building physics corpus...")
    text = build_corpus(args.n_lines, seed=args.seed)
    print(f"  corpus: {len(text):,} chars")
    print(f"  sample:\n{text[:300]}")
    vocab = CharVocab(text)
    print(f"  vocab size: {vocab.size}")

    # 80/20 train/val split.
    split = int(len(text) * 0.8)
    train_text = text[:split]
    val_text = text[split:]

    train_iter = make_batches(train_text, vocab, args.block_size, args.batch_size,
                              seed=args.seed)
    val_iter = make_batches(val_text, vocab, args.block_size, args.batch_size * 2,
                            seed=args.seed + 1)
    val_x, val_y = next(val_iter)

    bit = BitCharLM(vocab.size, d_model=args.d_model, n_heads=args.n_heads,
                    n_layers=args.n_layers, max_len=args.block_size)
    fp = FPCharLM(vocab.size, d_model=args.d_model, n_heads=args.n_heads,
                  n_layers=args.n_layers, max_len=args.block_size)
    print(f"\nBitCharLM params: {sum(p.numel() for p in bit.parameters()):,}")
    print(f"FPCharLM  params: {sum(p.numel() for p in fp.parameters()):,}\n")

    h_bit = train(bit, train_iter, val_x, val_y, args.steps, args.lr, "BitLM")
    print()
    train_iter = make_batches(train_text, vocab, args.block_size, args.batch_size,
                              seed=args.seed)  # reset
    h_fp = train(fp, train_iter, val_x, val_y, args.steps, args.lr, " FPLM")

    print("\n--- Sample generations from each (32 chars after a prompt) ---")
    prompts = ["oscillator: ", "pendulum: ", "schrodinger: "]
    for prompt in prompts:
        prompt_ids = torch.tensor([vocab.encode(prompt)], dtype=torch.long)
        bit_out = bit.generate(prompt_ids, 40, temperature=0.6)
        fp_out = fp.generate(prompt_ids, 40, temperature=0.6)
        print(f"\n  prompt: {prompt!r}")
        print(f"    Bit: {vocab.decode(bit_out[0].tolist())!r}")
        print(f"     FP: {vocab.decode(fp_out[0].tolist())!r}")

    print(f"\nBit final val ppl: {math.exp(h_bit['val'][-1]):.2f}")
    print(f"FP  final val ppl: {math.exp(h_fp['val'][-1]):.2f}")


if __name__ == "__main__":
    main()
