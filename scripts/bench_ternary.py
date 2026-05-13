"""Benchmark trit packing + multiply-free matmul vs fp32 baseline.

Run from project root: venv/bin/python -m scripts.bench_ternary
"""

from __future__ import annotations

import time

import numpy as np
import torch

from src.multfree import count_ops, matmul_split_masks
from src.trit_pack import (
    compression_ratio_vs_fp32,
    compression_ratio_vs_int8,
    pack_trits,
    storage_bytes,
    unpack_trits,
)


def time_call(fn, *args, repeat: int = 5) -> float:
    fn(*args)  # warmup
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best


def bench_storage():
    print("=" * 70)
    print("STORAGE — how many bytes does a ternary weight matrix take?")
    print("=" * 70)
    rng = np.random.default_rng(0)
    for shape in [(64, 64), (256, 256), (1024, 1024), (4096, 4096)]:
        n = shape[0] * shape[1]
        W = rng.integers(-1, 2, size=shape).astype(np.int8)
        packed, _ = pack_trits(W)
        fp32_bytes = n * 4
        int8_bytes = n
        ternary_bytes = storage_bytes(n)
        print(f"  {shape[0]}x{shape[1]} = {n:>10,} weights")
        print(f"      fp32:    {fp32_bytes:>12,} bytes")
        print(f"      int8:    {int8_bytes:>12,} bytes")
        print(f"      ternary: {ternary_bytes:>12,} bytes "
              f"({compression_ratio_vs_fp32(n):.1f}x vs fp32, "
              f"{compression_ratio_vs_int8(n):.1f}x vs int8)")

        # Verify round-trip on a sample
        sample = W.ravel()[:1000]
        s_packed, s_n = pack_trits(sample)
        assert np.array_equal(unpack_trits(s_packed, s_n), sample)
    print()


def bench_ops_skipped():
    print("=" * 70)
    print("COMPUTE — operation counts (one ternary matmul, 1024x1024)")
    print("=" * 70)
    rng = np.random.default_rng(0)
    W = rng.integers(-1, 2, size=(1024, 1024)).astype(np.int8)
    stats = count_ops(W)
    print(f"  total weights:        {stats['total_weights']:>12,}")
    print(f"  zeros (skipped):      {stats['total_weights'] - stats['nonzero_weights']:>12,} "
          f"({stats['skipped_fraction']:.1%})")
    print(f"  nonzero (add or sub): {stats['nonzero_weights']:>12,}")
    print()
    print(f"  fp32  matmul: {stats['fp32_multiplies']:>12,} multiplies "
          f"+ ~{stats['fp32_multiplies']:,} adds")
    print(f"  ternary mm:   {stats['ternary_multiplies']:>12,} multiplies "
          f"+ {stats['ternary_adds_subs']:,} signed adds")
    print()
    print("  -> the multiplier (the expensive part) is eliminated.")
    print("     in a real ASIC/FPGA, multipliers dominate area & energy.")
    print()


def bench_timing():
    print("=" * 70)
    print("TIMING — wall clock for forward pass (NumPy/Torch, CPU)")
    print("=" * 70)
    print("  Note: this measures the *semantic* operation; both end up in BLAS.")
    print("  The real win is hardware-level — see compute counts above.")
    print()
    rng = np.random.default_rng(0)
    sizes = [(256, 256), (512, 512), (1024, 1024), (2048, 2048)]
    for shape in sizes:
        W_int8 = rng.integers(-1, 2, size=shape).astype(np.int8)
        W_fp32 = W_int8.astype(np.float32)
        W_torch = torch.from_numpy(W_fp32)
        x_np = rng.standard_normal(shape[1]).astype(np.float32)
        x_torch = torch.from_numpy(x_np)

        t_np_fp32 = time_call(lambda: W_fp32 @ x_np)
        t_torch_fp32 = time_call(lambda: W_torch @ x_torch)
        t_ternary = time_call(lambda: matmul_split_masks(W_int8, x_np))

        print(f"  {shape[0]}x{shape[1]}:")
        print(f"      numpy fp32 matmul: {t_np_fp32*1000:7.2f} ms")
        print(f"      torch fp32 matmul: {t_torch_fp32*1000:7.2f} ms")
        print(f"      ternary mult-free: {t_ternary*1000:7.2f} ms")
    print()


def bench_end_to_end_memory():
    print("=" * 70)
    print("END-TO-END — a BitMLP's hidden weights compressed")
    print("=" * 70)
    # Match the train.py config: 5 layers, hidden=128, of which 3 are BitLinear
    # (input proj and output head stay fp32). Each BitLinear is 128x128 + bias.
    n_bitlinear_layers = 3
    hidden_weights = n_bitlinear_layers * 128 * 128
    fp32_bytes = hidden_weights * 4
    ternary_bytes = storage_bytes(hidden_weights)
    print(f"  3 hidden BitLinear layers, 128x128 each = {hidden_weights:,} weights")
    print(f"      fp32:    {fp32_bytes:>10,} bytes ({fp32_bytes/1024:.1f} KB)")
    print(f"      ternary: {ternary_bytes:>10,} bytes "
          f"({ternary_bytes/1024:.2f} KB) — {fp32_bytes/ternary_bytes:.1f}x smaller")
    print()
    print("  Imagine the same ratio on a 7B-parameter LLM:")
    big = 7_000_000_000
    print(f"      fp32:    {big*4/1e9:7.2f} GB")
    print(f"      ternary: {storage_bytes(big)/1e9:7.2f} GB "
          f"({(big*4)/storage_bytes(big):.1f}x smaller)")
    print()


if __name__ == "__main__":
    bench_storage()
    bench_ops_skipped()
    bench_timing()
    bench_end_to_end_memory()
