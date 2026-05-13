"""Train the ternary trajectory transformer on damped oscillator sequences.

Given a prefix of x(t_0), x(t_1), ..., x(t_{N-1}), predict the next step.
At inference we roll out autoregressively for `gen_steps` steps and compare
to the true trajectory.

Run: venv/bin/python -m src.train_transformer [--plot]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn

from .data import damped_oscillator
from .transformer import BitTrajectoryTransformer, FPTrajectoryTransformer


def make_trajectories(n: int, seq_len: int, seed: int) -> torch.Tensor:
    """Generate n damped-oscillator trajectories of length seq_len."""
    g = torch.Generator().manual_seed(seed)
    omega = 0.5 + torch.rand(n, generator=g) * 2.5
    zeta = 0.05 + torch.rand(n, generator=g) * 0.4
    dt = 0.1
    t = torch.arange(seq_len) * dt
    # Broadcast: (n,1) and (seq_len,) -> (n, seq_len)
    t_b = t.unsqueeze(0).expand(n, seq_len)
    omega_b = omega.unsqueeze(1).expand(n, seq_len)
    zeta_b = zeta.unsqueeze(1).expand(n, seq_len)
    x = damped_oscillator(t_b, omega_b, zeta_b)
    return x.unsqueeze(-1)  # (n, seq_len, 1)


def train(model, data_train, data_val, epochs, lr, batch_size, label):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    n = data_train.shape[0]
    history = {"train": [], "val": []}
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            batch = data_train[idx]
            x_in = batch[:, :-1, :]
            x_target = batch[:, 1:, :]
            opt.zero_grad()
            loss = loss_fn(model(x_in), x_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * idx.shape[0]
        sched.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(data_val[:, :-1]), data_val[:, 1:]).item()
        history["train"].append(total / n)
        history["val"].append(val_loss)
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            print(f"[{label}] epoch {epoch+1:3d}/{epochs}  "
                  f"train={total/n:.5f}  val={val_loss:.5f}")
    print(f"[{label}] done in {time.time()-t0:.1f}s")
    return history


def rollout_eval(model, data, prefix_len, gen_steps, label):
    model.eval()
    prefix = data[:, :prefix_len]
    target = data[:, prefix_len:prefix_len + gen_steps]
    generated = model.generate(prefix, gen_steps)
    pred = generated[:, prefix_len:prefix_len + gen_steps]
    err = (pred - target).abs().mean().item()
    print(f"[{label}] autoregressive rollout ({gen_steps} steps) MAE: {err:.5f}")
    return generated, err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--prefix-len", type=int, default=16)
    parser.add_argument("--gen-steps", type=int, default=48)
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-val", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    data_train = make_trajectories(args.n_train, args.seq_len, args.seed)
    data_val = make_trajectories(args.n_val, args.seq_len, args.seed + 1)

    print(f"Trajectory shape: {tuple(data_train.shape)}")
    print(f"Value range: [{data_train.min():.2f}, {data_train.max():.2f}]\n")

    bit = BitTrajectoryTransformer(
        d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, max_len=args.seq_len,
    )
    fp = FPTrajectoryTransformer(
        d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, max_len=args.seq_len,
    )
    bit_params = sum(p.numel() for p in bit.parameters())
    fp_params = sum(p.numel() for p in fp.parameters())
    print(f"Bit-Transformer params: {bit_params:,}")
    print(f"FP-Transformer params:  {fp_params:,}\n")

    hist_bit = train(bit, data_train, data_val, args.epochs, args.lr, args.batch_size,
                     "BitTx")
    print()
    hist_fp = train(fp, data_train, data_val, args.epochs, args.lr, args.batch_size,
                    " FPTx")

    print()
    bit_gen, bit_mae = rollout_eval(bit, data_val, args.prefix_len, args.gen_steps, "BitTx")
    fp_gen, fp_mae = rollout_eval(fp, data_val, args.prefix_len, args.gen_steps, " FPTx")

    print(f"\nRatio (BitTx/FPTx) rollout MAE: {bit_mae/fp_mae:.2f}x")

    if args.plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 2, figsize=(14, 8))

        ax[0, 0].plot(hist_bit["val"], label="BitTx (ternary)")
        ax[0, 0].plot(hist_fp["val"], label="FPTx (fp32)")
        ax[0, 0].set_yscale("log")
        ax[0, 0].set_xlabel("epoch")
        ax[0, 0].set_ylabel("val MSE (next-step)")
        ax[0, 0].set_title("Training curves")
        ax[0, 0].legend()

        for k, idx in enumerate([0, 5, 17]):
            if k >= 3:
                break
            axi = [ax[0, 1], ax[1, 0], ax[1, 1]][k]
            t = np.arange(args.seq_len) * 0.1
            true_seq = data_val[idx, :, 0].numpy()
            bit_seq = bit_gen[idx, :, 0].numpy()
            fp_seq = fp_gen[idx, :, 0].numpy()
            axi.plot(t, true_seq, "k-", label="true", linewidth=2)
            axi.plot(t[:args.prefix_len], true_seq[:args.prefix_len], "go",
                     markersize=4, label="prefix (given)")
            axi.plot(t[args.prefix_len:args.prefix_len+args.gen_steps],
                     bit_seq[args.prefix_len:args.prefix_len+args.gen_steps],
                     "b--", label="BitTx rollout")
            axi.plot(t[args.prefix_len:args.prefix_len+args.gen_steps],
                     fp_seq[args.prefix_len:args.prefix_len+args.gen_steps],
                     "r:", label="FPTx rollout")
            axi.axvline(t[args.prefix_len], color="gray", linestyle=":", alpha=0.5)
            axi.set_xlabel("t")
            axi.set_ylabel("x(t)")
            axi.set_title(f"trajectory #{idx}")
            axi.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig("transformer_results.png", dpi=120)
        print("\nSaved plot to transformer_results.png")


if __name__ == "__main__":
    main()
