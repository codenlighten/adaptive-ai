"""STE-free ternary training via discrete coordinate-descent.

The Straight-Through Estimator is a clever hack: in forward we use discrete
weights, in backward we pretend they're continuous. It works well, but it
*lies* about the gradients. A discrete model's true loss surface is
piecewise constant; gradients don't really exist.

Discrete coordinate descent does the honest thing: cycle through weights
one at a time, try each of {-1, 0, +1} for the current weight, pick the
value that lowers loss the most, repeat. No gradients at all. Slower
per step but no STE bias.

This module implements that for a single-layer ternary linear regression
(supervised by least-squares loss). On a problem where the truth is
literally a ternary combination of features, this should *exactly*
recover the ground truth — no STE noise.
"""

from __future__ import annotations

import numpy as np


def discrete_coordinate_descent(
    X: np.ndarray,                       # (n, d) features
    y: np.ndarray,                       # (n,) targets
    levels: tuple[int, ...] = (-1, 0, 1),
    max_passes: int = 30,
    init: np.ndarray | None = None,
    alpha_init: float = 1.0,
) -> tuple[np.ndarray, float, list[float]]:
    """Find w in `levels`^d and a scale alpha that minimize ||y - alpha * X @ w||^2.

    Strategy: alternate between
       1. fix w, fit alpha via closed-form least squares (1-D scalar)
       2. fix alpha, do one pass of discrete coordinate descent on w
    until no weights change (or max_passes reached).

    No STE. No gradients. Just exhaustive per-coordinate evaluation.
    """
    n, d = X.shape
    if init is None:
        # Start at sign(X^T y) — best constant ternary guess from correlation.
        c = X.T @ y
        # threshold at half the max abs corr
        thr = 0.5 * np.abs(c).mean()
        w = np.where(np.abs(c) < thr, 0, np.sign(c)).astype(np.int64)
    else:
        w = init.astype(np.int64).copy()

    alpha = float(alpha_init)
    losses: list[float] = []

    def current_loss(alpha_v, w_v):
        return float(((y - alpha_v * (X @ w_v.astype(np.float64))) ** 2).mean())

    losses.append(current_loss(alpha, w))

    for it in range(max_passes):
        # 1. fit alpha given w
        Xw = X @ w.astype(np.float64)
        denom = float((Xw ** 2).sum())
        alpha = float((Xw * y).sum() / denom) if denom > 1e-12 else 0.0

        # 2. discrete coordinate descent on w
        changed = False
        order = np.random.permutation(d)
        for j in order:
            best_v = int(w[j])
            best_loss = current_loss(alpha, w)
            for v in levels:
                if v == best_v:
                    continue
                w[j] = v
                l = current_loss(alpha, w)
                if l < best_loss:
                    best_loss = l
                    best_v = v
            if best_v != int(w[j]):
                changed = True
            w[j] = best_v

        losses.append(current_loss(alpha, w))
        if not changed:
            break

    return w, alpha, losses
