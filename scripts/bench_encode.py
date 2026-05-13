"""Microbenchmark: vectorized vs loop vs fast-mean encoders.

Measures the per-call latency of the three encode paths across realistic
weight shapes and T values. Run on CPU; pass --cuda to compare on GPU.

Usage:
    python -m scripts.bench_encode
    python -m scripts.bench_encode --cuda

Reference results (CPU, x86_64, fp32, 50 iters median):

    shape           T   loop (ms)  vec (ms)  mean (ms)  vec   mean
    (256, 256)      8     3.51       0.94     0.18     3.8x  19.5x
    (256, 256)     32    14.61       4.77     0.18     3.1x  79.7x
    (1024, 1024)    8    24.41      37.65     1.15     0.7x  21.2x
    (4096, 1024)    8   164.48     157.78    13.54     1.0x  12.2x

The full vec encoder helps most for small shapes (where Python loop
overhead dominates). The fast-mean path (`delta_sigma_mean_ternary`)
strictly wins everywhere — it never materializes the (T, *shape) stream.
This is the path the training forward uses; the resulting end-to-end
training-step speedup is 4-13x at typical sizes (T=8-32).
"""

from __future__ import annotations

import argparse
import time

import torch

from delta_sigma_nn.delta_sigma import (
    _encode_ternary_loop,
    delta_sigma_mean_ternary,
    encode_delta_sigma_ternary,
)


def _bench(fn, *args, iters: int, warmup: int = 5, device: str = "cpu") -> float:
    """Return median per-call seconds."""
    for _ in range(warmup):
        out = fn(*args)
    if device == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn(*args)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    del out
    times.sort()
    return times[len(times) // 2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true",
                        help="run on CUDA instead of CPU")
    parser.add_argument("--iters", type=int, default=50,
                        help="median over this many calls")
    args = parser.parse_args()

    device = "cuda" if args.cuda else "cpu"
    if args.cuda and not torch.cuda.is_available():
        raise SystemExit("--cuda requested but torch.cuda is unavailable")

    shapes = [(256, 256), (1024, 1024), (4096, 1024)]
    Ts = [4, 8, 16, 32]

    print(f"device: {device}")
    print(f"iters per measurement: {args.iters} (median)")
    print()
    header = f"{'shape':<15}{'T':>4}  {'loop (ms)':>12}{'vec (ms)':>12}{'mean (ms)':>12}  {'vec speedup':>12}{'mean speedup':>13}"
    print(header)
    print("-" * len(header))

    for shape in shapes:
        for T in Ts:
            torch.manual_seed(0)
            W = (torch.rand(shape, device=device) * 2 - 1) * 0.9
            t_loop = _bench(_encode_ternary_loop, W, T, iters=args.iters, device=device)
            t_vec  = _bench(encode_delta_sigma_ternary, W, T, iters=args.iters, device=device)
            t_mean = _bench(delta_sigma_mean_ternary,  W, T, iters=args.iters, device=device)
            print(f"{str(shape):<15}{T:>4}  "
                  f"{t_loop*1e3:>12.3f}{t_vec*1e3:>12.3f}{t_mean*1e3:>12.3f}  "
                  f"{t_loop/t_vec:>11.2f}x{t_loop/t_mean:>12.2f}x")


if __name__ == "__main__":
    main()
