"""Train a DSigma transformer on the physics corpus.

Compare against BitCharLM (ternary) and FPCharLM at matched parameter count.

Run: venv/bin/python -m scripts.train_dsigma_transformer
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.char_lm import BitCharLM, FPCharLM
from src.dsigma_transformer import DSigmaCharLM
from src.physics_corpus import CharVocab, build_corpus, make_batches


def train(model, batch_iter, val_x, val_y, steps, lr, label):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
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
        if step % max(1, steps // 8) == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                vl = F.cross_entropy(
                    model(val_x).reshape(-1, model.head.out_features),
                    val_y.reshape(-1),
                ).item()
            print(f"[{label}] step {step+1:5d}/{steps}  val={vl:.3f}  ppl={math.exp(vl):.2f}")
    print(f"[{label}] done in {time.time()-t0:.1f}s")
    model.eval()
    with torch.no_grad():
        return F.cross_entropy(
            model(val_x).reshape(-1, model.head.out_features),
            val_y.reshape(-1),
        ).item()


def main():
    torch.manual_seed(0)
    text = build_corpus(6000, seed=0)
    vocab = CharVocab(text)
    print(f"Corpus: {len(text):,} chars  Vocab: {vocab.size}\n")
    split = int(len(text) * 0.9)
    train_text, val_text = text[:split], text[split:]
    val_iter = make_batches(val_text, vocab, 64, 128, seed=100)
    val_x, val_y = next(val_iter)

    configs = [
        ("BitTx (ternary)", lambda: BitCharLM(vocab.size, 128, 4, 4, max_len=64)),
        ("DSigma T=4",      lambda: DSigmaCharLM(vocab.size, 128, 4, 4, max_len=64, T=4)),
        ("DSigma T=8",      lambda: DSigmaCharLM(vocab.size, 128, 4, 4, max_len=64, T=8)),
        ("DSigma T=16",     lambda: DSigmaCharLM(vocab.size, 128, 4, 4, max_len=64, T=16)),
        ("FPTx (fp32)",     lambda: FPCharLM(vocab.size, 128, 4, 4, max_len=64)),
    ]
    results = []
    for label, make in configs:
        torch.manual_seed(0)
        model = make()
        n_params = sum(p.numel() for p in model.parameters())
        train_iter = make_batches(train_text, vocab, 64, 64, seed=0)
        val_loss = train(model, train_iter, val_x, val_y, 1000, 2e-3, label)
        results.append({"label": label, "n_params": n_params, "val_loss": val_loss,
                        "ppl": math.exp(val_loss)})

    print("\n=== Comparison ===")
    print(f"{'config':>20}  {'params':>9}  {'val_ppl':>9}")
    for r in results:
        print(f"{r['label']:>20}  {r['n_params']:>9,}  {r['ppl']:>9.3f}")


if __name__ == "__main__":
    main()
