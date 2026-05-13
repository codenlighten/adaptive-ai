"""Train a larger DSigma transformer with BPE tokens.

Targets ~5M parameters at 8 layers × 256 d_model × 8 heads. Tests
whether the small-scale fp32-beating result generalizes at 10× scale.

Run: venv/bin/python -m scripts.scale_dsigma_lm
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.bpe import BPETokenizer
from src.char_lm import BitCharLM, FPCharLM
from src.dsigma_transformer import DSigmaCharLM
from src.physics_corpus import build_corpus


def make_batches_tokens(token_ids, block_size, batch_size, seed=0):
    data = torch.tensor(token_ids, dtype=torch.long)
    n = data.shape[0]
    g = torch.Generator().manual_seed(seed)
    while True:
        ix = torch.randint(0, n - block_size - 1, (batch_size,), generator=g)
        x = torch.stack([data[i:i + block_size] for i in ix])
        y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
        yield x, y


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
    text = build_corpus(15000, seed=0)
    print(f"Corpus: {len(text):,} chars")

    tok = BPETokenizer.train(text, vocab_size=384)
    ids = tok.encode(text)
    print(f"BPE vocab {tok.vocab_size}, tokens {len(ids):,} "
          f"(compression {len(text)/len(ids):.2f}x)\n")

    split = int(len(ids) * 0.9)
    train_ids, val_ids = ids[:split], ids[split:]
    block_size = 96
    val_iter = make_batches_tokens(val_ids, block_size, 96, seed=100)
    val_x, val_y = next(val_iter)

    # 8-layer 256-d transformer ≈ ~3-5M params (vocab adds embedding mass)
    common = dict(d_model=256, n_heads=8, n_layers=6, max_len=block_size)
    configs = [
        ("FullBit",     lambda: BitCharLM(tok.vocab_size, **common)),
        ("DSigma T=8",  lambda: DSigmaCharLM(tok.vocab_size, **common, T=8)),
        ("FP32",        lambda: FPCharLM(tok.vocab_size, **common)),
    ]
    results = []
    for label, make in configs:
        torch.manual_seed(0)
        model = make()
        n_params = sum(p.numel() for p in model.parameters())
        train_iter = make_batches_tokens(train_ids, block_size, 32, seed=0)
        val_loss = train(model, train_iter, val_x, val_y, 1200, 1.5e-3, label)
        results.append({"label": label, "n_params": n_params,
                        "val_loss": val_loss, "ppl": math.exp(val_loss)})

    print("\n=== Comparison ===")
    print(f"{'config':>14}  {'params':>10}  {'val_ppl':>9}")
    for r in results:
        print(f"{r['label']:>14}  {r['n_params']:>10,}  {r['ppl']:>9.3f}")

    # Save the DSigma model for the router experiments
    ds_model = configs[1][1]()
    torch.manual_seed(0)
    train_iter = make_batches_tokens(train_ids, block_size, 32, seed=0)
    train(ds_model, train_iter, val_x, val_y, 1200, 1.5e-3, "DSigma-save")
    torch.save({
        "state_dict": ds_model.state_dict(),
        "vocab_size": tok.vocab_size,
        "T": 8,
        "block_size": block_size,
        "vocab_chars": tok.vocab,
        "merges": tok.merges,
    }, "scaled_dsigma_model.pt")
    print("\nSaved scaled_dsigma_model.pt for router experiments")


if __name__ == "__main__":
    main()
