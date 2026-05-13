"""Train a ternary LM on BPE-tokenized physics text.

BPE compresses repeated patterns ('omega=', 'kepler:', ':+0', etc.) into
single tokens, giving the LM a much longer effective context for the same
sequence length. Tests whether the ternary-matches-fp story holds at the
token level (more LLM-realistic than char-level).

Run: venv/bin/python -m scripts.train_bpe_lm
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.bpe import BPETokenizer
from src.char_lm import BitCharLM, FPCharLM
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
        if step % max(1, steps // 10) == 0 or step == steps - 1:
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
    text = build_corpus(8000, seed=0)
    print(f"Corpus: {len(text):,} chars\n")

    print("Training BPE tokenizer (vocab=512)...")
    tok = BPETokenizer.train(text, vocab_size=512)
    print(f"  final vocab size: {tok.vocab_size}")
    ids = tok.encode(text)
    print(f"  tokenized length: {len(ids):,} tokens  "
          f"(compression {len(text)/len(ids):.2f}x)")
    print(f"  longest token by bytes: "
          f"{tok.vocab[max(range(tok.vocab_size), key=lambda i: len(tok.vocab[i]))]!r}")

    # Show some longest tokens — the model has learned interesting subwords
    sorted_vocab = sorted(enumerate(tok.vocab), key=lambda x: -len(x[1]))
    print("  top-10 longest tokens:", [v.decode("utf-8", "replace") for _, v in sorted_vocab[:10]])

    split = int(len(ids) * 0.9)
    train_ids, val_ids = ids[:split], ids[split:]

    block_size = 64
    train_iter = make_batches_tokens(train_ids, block_size, batch_size=48, seed=0)
    val_iter = make_batches_tokens(val_ids, block_size, batch_size=96, seed=100)
    val_x, val_y = next(val_iter)

    bit = BitCharLM(tok.vocab_size, d_model=128, n_heads=4, n_layers=4, max_len=block_size)
    fp = FPCharLM(tok.vocab_size, d_model=128, n_heads=4, n_layers=4, max_len=block_size)

    print(f"\nBitCharLM params: {sum(p.numel() for p in bit.parameters()):,}")
    print(f"FPCharLM  params: {sum(p.numel() for p in fp.parameters()):,}\n")

    bit_loss = train(bit, train_iter, val_x, val_y, 1500, 2e-3, "BitLM-BPE")
    print()
    train_iter = make_batches_tokens(train_ids, block_size, batch_size=48, seed=0)
    fp_loss = train(fp, train_iter, val_x, val_y, 1500, 2e-3, " FPLM-BPE")

    print("\n=== Generation samples ===")
    prompts = ["oscillator: ", "pendulum: ", "kepler: "]
    for prompt in prompts:
        prompt_ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
        bit_out = bit.generate(prompt_ids, 40, temperature=0.5)
        fp_out = fp.generate(prompt_ids, 40, temperature=0.5)
        print(f"\n  prompt: {prompt!r}")
        print(f"    Bit: {tok.decode(bit_out[0].tolist())!r}")
        print(f"     FP: {tok.decode(fp_out[0].tolist())!r}")

    print(f"\nBit ppl: {math.exp(bit_loss):.3f}")
    print(f"FP  ppl: {math.exp(fp_loss):.3f}")
    print(f"ratio:   {math.exp(bit_loss)/math.exp(fp_loss):.4f}")


if __name__ == "__main__":
    main()
