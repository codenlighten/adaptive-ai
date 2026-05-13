"""FGSM and PGD input attacks on BitMLP vs FPMLP.

Both networks are trained on the damped oscillator. We then craft input
perturbations that maximize MSE under a budget of L-infinity radius epsilon.
The question: does ternary's discrete weight space make the loss
landscape more or less brittle to such attacks?

FGSM: x' = x + eps * sign(grad_x loss)
PGD:  iterate FGSM steps with smaller alpha and project back to ||delta||_inf <= eps
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

from src.data import make_dataset, normalize
from src.model import BitMLP, FPMLP


def fgsm_attack(model, x, y, epsilon, loss_fn):
    x_adv = x.clone().detach().requires_grad_(True)
    pred = model(x_adv)
    loss = loss_fn(pred, y)
    grad = torch.autograd.grad(loss, x_adv)[0]
    return (x_adv + epsilon * grad.sign()).detach()


def pgd_attack(model, x, y, epsilon, alpha, n_steps, loss_fn):
    x_orig = x.clone().detach()
    x_adv = x.clone().detach() + torch.empty_like(x).uniform_(-epsilon, epsilon)
    for _ in range(n_steps):
        x_adv = x_adv.detach().requires_grad_(True)
        pred = model(x_adv)
        loss = loss_fn(pred, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv + alpha * grad.sign()
        delta = (x_adv - x_orig).clamp(-epsilon, epsilon)
        x_adv = (x_orig + delta).detach()
    return x_adv


def evaluate(model, x, y):
    loss_fn = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        return loss_fn(model(x), y).item()


def train(model, X, y, X_val, y_val, epochs, lr, batch_size, label):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    n = X.shape[0]
    t0 = time.time()
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss_fn(model(X[idx]), y[idx]).backward()
            opt.step()
        sched.step()
    print(f"  [{label}] trained in {time.time()-t0:.1f}s, val MSE = {evaluate(model, X_val, y_val):.6f}")


def main():
    torch.manual_seed(0)
    X_train, y_train = make_dataset(8000, seed=0)
    X_val, y_val = make_dataset(2000, seed=1)
    Xtn, stats = normalize(X_train)
    Xvn, _ = normalize(X_val, stats)

    print("Training BitMLP and FPMLP on damped oscillator...")
    bit = BitMLP(3, 128, 1, depth=5)
    fp = FPMLP(3, 128, 1, depth=5)
    train(bit, Xtn, y_train, Xvn, y_val, epochs=200, lr=2e-3, batch_size=256, label="bit")
    train(fp, Xtn, y_train, Xvn, y_val, epochs=200, lr=2e-3, batch_size=256, label=" fp")
    bit.eval(); fp.eval()

    loss_fn = nn.MSELoss()
    epsilons = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

    print("\n=== FGSM attack — input MSE under L-infty perturbation budget ===")
    print(f"{'epsilon':>10}  {'Bit MSE':>12}  {'FP MSE':>12}  {'Bit/FP':>8}")
    fgsm_results = []
    for eps in epsilons:
        if eps == 0.0:
            bit_l = evaluate(bit, Xvn, y_val)
            fp_l = evaluate(fp, Xvn, y_val)
        else:
            X_bit_adv = fgsm_attack(bit, Xvn, y_val, eps, loss_fn)
            X_fp_adv  = fgsm_attack(fp,  Xvn, y_val, eps, loss_fn)
            bit_l = evaluate(bit, X_bit_adv, y_val)
            fp_l  = evaluate(fp,  X_fp_adv,  y_val)
        ratio = bit_l / max(fp_l, 1e-12)
        fgsm_results.append((eps, bit_l, fp_l, ratio))
        print(f"{eps:>10.3f}  {bit_l:>12.6f}  {fp_l:>12.6f}  {ratio:>7.2f}x")

    print("\n=== PGD attack (10 steps, alpha = eps/4) ===")
    print(f"{'epsilon':>10}  {'Bit MSE':>12}  {'FP MSE':>12}  {'Bit/FP':>8}")
    for eps in epsilons:
        if eps == 0.0:
            bit_l = evaluate(bit, Xvn, y_val)
            fp_l = evaluate(fp, Xvn, y_val)
        else:
            X_bit_adv = pgd_attack(bit, Xvn, y_val, eps, eps / 4.0, 10, loss_fn)
            X_fp_adv  = pgd_attack(fp,  Xvn, y_val, eps, eps / 4.0, 10, loss_fn)
            bit_l = evaluate(bit, X_bit_adv, y_val)
            fp_l  = evaluate(fp,  X_fp_adv,  y_val)
        ratio = bit_l / max(fp_l, 1e-12)
        print(f"{eps:>10.3f}  {bit_l:>12.6f}  {fp_l:>12.6f}  {ratio:>7.2f}x")

    print("\n=== Interpretation ===")
    print("  - ratio < 1: ternary is MORE robust (smaller adversarial degradation)")
    print("  - ratio > 1: fp32 is more robust")
    print("  - At eps=0 both perform at clean test MSE")


if __name__ == "__main__":
    main()
