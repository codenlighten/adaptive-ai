"""Capstone^2: Run an ENTIRE trained BitMLP through the Setun VM end-to-end.

The matmul of every BitLinear layer is compiled to Setun instructions and
executed on the balanced-ternary VM (with `Trit18` digit arithmetic and no
multiplications). Between layers, the host handles things that are
fundamentally irrational and not expressible in finite balanced-ternary
without a lookup table: LayerNorm (sqrt) and GELU (erf). A real Setun
deployment would do these via approximation tables or fixed-point series;
here we run them in NumPy and account for the residual error.

This closes the *full* loop: an end-to-end forward pass of a network
trained with Adam on a damped harmonic oscillator, executed by a
simulated 1958-style ternary computer.

Run: venv/bin/python -m scripts.run_setun_full
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.data import make_dataset, normalize
from src.model import BitMLP
from src.setun import Trit18, matmul_as_setun_program
from src.ternary import BitLinear, ternarize


# ---------------------------------------------------------------------------
# Host-side helpers (LayerNorm + GELU). These are NOT in ternary because
# 1/sqrt and erf are irrational; in a Setun deployment they'd come from
# a lookup table. We're being honest about this.
# ---------------------------------------------------------------------------

def layernorm_host(x: np.ndarray, w: np.ndarray, b: np.ndarray,
                   eps: float = 1e-5) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    return ((x - mu) / np.sqrt(var + eps)) * w + b


def gelu_host(x: np.ndarray) -> np.ndarray:
    from scipy.special import erf
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


# ---------------------------------------------------------------------------
# Run one BitLinear forward pass through the SetunVM.
# ---------------------------------------------------------------------------

def bitlinear_on_setun(x: np.ndarray, layer: BitLinear, scale: int = 1000,
                       stats: dict | None = None) -> tuple[np.ndarray, dict]:
    """Compute y = BitLinear(x) using SetunVM for the matmul.

    Returns (y, stats_delta).
    """
    if stats is None:
        stats = {"add": 0, "sub": 0, "mul": 0, "trit_full_adds": 0, "matmul_ops": 0}

    with torch.no_grad():
        w_q, alpha = ternarize(layer.weight)
        W = w_q.numpy().astype(int).tolist()
        alpha = float(alpha)
        bias = layer.bias.detach().numpy() if layer.bias is not None else None
        ln_w = layer.norm.weight.detach().numpy()
        ln_b = layer.norm.bias.detach().numpy()

    # Step 1: LayerNorm (host).
    x_normed = layernorm_host(x, ln_w, ln_b)  # shape (B, in)

    out_dim = len(W)
    in_dim = len(W[0])
    batch = x_normed.shape[0]
    y = np.zeros((batch, out_dim), dtype=np.float32)

    # Step 2: per-batch-element, quantize input and run Setun matmul.
    for b_idx in range(batch):
        x_int = (x_normed[b_idx] * scale).round().astype(int).tolist()
        program, vm = matmul_as_setun_program(W, x_int)
        vm.run(program)
        y_int = np.array([vm.mem[in_dim + i].to_int() for i in range(out_dim)])
        y[b_idx] = alpha * y_int / scale
        # Aggregate stats over the whole pass.
        stats["add"] += vm.stats["add"]
        stats["sub"] += vm.stats["sub"]
        stats["mul"] += vm.stats["mul"]
        stats["trit_full_adds"] += vm.stats["trit_full_adds"]
        stats["matmul_ops"] += 1

    if bias is not None:
        y = y + bias
    return y, stats


# ---------------------------------------------------------------------------
# Full forward pass.
# ---------------------------------------------------------------------------

def forward_full_on_setun(model: BitMLP, x: np.ndarray,
                          scale: int = 1000) -> tuple[np.ndarray, dict]:
    """Run a full BitMLP forward pass with all BitLinear matmuls on SetunVM."""
    h = x.astype(np.float32)
    if h.ndim == 1:
        h = h[None, :]
        squeeze = True
    else:
        squeeze = False

    stats = {"add": 0, "sub": 0, "mul": 0, "trit_full_adds": 0, "matmul_ops": 0}

    # Walk the network in order.
    for mod in model.net:
        if isinstance(mod, BitLinear):
            h, stats = bitlinear_on_setun(h, mod, scale=scale, stats=stats)
        elif isinstance(mod, nn.Linear):
            with torch.no_grad():
                W = mod.weight.detach().numpy()
                b = mod.bias.detach().numpy() if mod.bias is not None else None
            h = h @ W.T
            if b is not None:
                h = h + b
        elif isinstance(mod, nn.GELU):
            h = gelu_host(h)
        else:
            raise TypeError(f"unsupported module {type(mod)}")

    return (h.squeeze(0) if squeeze else h), stats


# ---------------------------------------------------------------------------
# Demo.
# ---------------------------------------------------------------------------

def train_quick():
    torch.manual_seed(0)
    X_train, y_train = make_dataset(2000, seed=0)
    X_val, y_val = make_dataset(200, seed=1)
    X_train_n, st = normalize(X_train)
    X_val_n, _ = normalize(X_val, st)

    # Small model to keep Setun runtime reasonable.
    model = BitMLP(3, 32, 1, depth=4)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
    loss_fn = nn.MSELoss()
    bs = 256
    for _ in range(100):
        model.train()
        perm = torch.randperm(X_train_n.shape[0])
        for i in range(0, X_train_n.shape[0], bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss_fn(model(X_train_n[idx]), y_train[idx]).backward()
            opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        torch_mse = loss_fn(model(X_val_n), y_val).item()
    return model, X_val_n, y_val, torch_mse


def main():
    print("=" * 70)
    print("Train a small BitMLP (32-wide, depth-4) on the damped oscillator")
    print("=" * 70)
    model, X_val_n, y_val, torch_mse = train_quick()
    print(f"  torch val MSE: {torch_mse:.6f}")

    # Pick a small batch — Setun is slow.
    n_eval = 32
    x = X_val_n[:n_eval].numpy()
    y_true = y_val[:n_eval].numpy()

    print()
    print("=" * 70)
    print(f"Run full network forward on Setun VM (batch={n_eval})")
    print("=" * 70)
    import time
    t0 = time.time()
    y_setun, stats = forward_full_on_setun(model, x, scale=10_000)
    print(f"  Setun run time: {time.time()-t0:.1f}s")

    with torch.no_grad():
        y_torch = model(X_val_n[:n_eval]).numpy()

    diff = np.abs(y_setun - y_torch)
    print()
    print(f"  max |y_setun - y_torch|:  {diff.max():.6e}")
    print(f"  mean |y_setun - y_torch|: {diff.mean():.6e}")

    setun_mse = ((y_setun - y_true) ** 2).mean()
    torch_subset_mse = ((y_torch - y_true) ** 2).mean()
    print(f"\n  torch BitMLP MSE on same batch: {torch_subset_mse:.6f}")
    print(f"  Setun-VM   BitMLP MSE on same batch: {float(setun_mse):.6f}")

    print()
    print("=" * 70)
    print("Operation counts across the entire forward pass")
    print("=" * 70)
    print(f"  ternary-matmul invocations:   {stats['matmul_ops']:>12,}")
    print(f"  total ADDs:                    {stats['add']:>12,}")
    print(f"  total SUBs:                    {stats['sub']:>12,}")
    print(f"  total MULs:                    {stats['mul']:>12,}  (must be 0)")
    print(f"  ternary digit full-adders:     {stats['trit_full_adds']:>12,}")
    assert stats["mul"] == 0, "matmul on Setun must use zero multiplications"

    print()
    print("-" * 70)
    print("A trained neural network for damped harmonic oscillator dynamics")
    print("just executed entirely on a simulated balanced-ternary computer.")
    print("Every matrix multiplication was reduced to signed addition of")
    print("18-trit balanced ternary words. The only non-ternary parts —")
    print("LayerNorm, GELU, and the small boundary projections — would, in")
    print("a real Setun deployment, be implemented via lookup tables.")


if __name__ == "__main__":
    main()
