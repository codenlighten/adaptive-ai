"""Cycle-accurate Setun program for ternary matmul.

Where `run_setun.py` generates one unrolled program per matmul, here we
write a SINGLE Setun program with control flow (loops, conditional
branches, indexed memory access) that walks over weights and inputs
itself. This is what a real program for the 1958 Setun would have
looked like, structurally — code in memory, branches, register-based
indexing.

Memory layout:
  mem[0]                = m (output dim)
  mem[1]                = n (input dim)
  mem[2 .. 2+n-1]       = x_0 .. x_{n-1}              (inputs)
  mem[X_BASE+i*n+j]     = W[i,j] in {-1, 0, +1}       (weights, row-major)
  mem[Y_BASE+i]         = y_i (outputs, written by program)

Registers:
  R0 = accumulator (sum for current output row)
  R1 = i (output row counter)
  R2 = j (input column counter)
  R3 = weight value W[i,j]
  R4 = x_j scratch
  R5 = address scratch
  R6 = constant 1 (for increments)
  R7 = m (loop bound) / n (loop bound)

Pseudocode:
  for i in 0..m-1:
      R0 = 0
      for j in 0..n-1:
          R3 = W[i,j]
          if R3 != 0:
              R4 = x_j
              if R3 > 0: R0 += R4
              else:      R0 -= R4
          j += 1
      y_i = R0
      i += 1
  HLT

This program structure has just one matmul kernel for all (m, n) — the
entire matrix multiply runs as one Setun program with branches.
"""

from __future__ import annotations

import torch

from src.model import BitMLP
from src.setun import SetunVM, Trit18
from src.ternary import BitLinear, ternarize


def build_matmul_program(m: int, n: int, X_BASE: int, Y_BASE: int) -> list[tuple]:
    """Return the Setun program (label-resolved at run-time) for y = W @ x.

    No multiplies, two nested loops, control flow via JNZ/JP/JN."""
    return [
        # --- outer loop init ---
        ("LDI", 1, 0),                       # R1 (i) <- 0
        ("LDI", 6, 1),                       # R6 <- 1 (increment constant)
        ("LABEL", "outer_top"),
        ("LDI", 7, m),                       # R7 <- m  (reloaded each iteration)
        ("SUB",  0, 1, 7),                   # R0 <- i - m
        ("JZ",   0, "outer_done"),           # if i == m: exit
        # --- inner loop init ---
        ("LDI", 0, 0),                       # R0 <- 0 (accumulator)
        ("LDI", 2, 0),                       # R2 (j) <- 0
        ("LDI", 7, n),                       # R7 <- n
        ("LABEL", "inner_top"),
        ("SUB",  3, 2, 7),                   # R3 <- j - n
        ("JZ",   3, "inner_done"),           # if j == n: row done
        # weight addr = X_BASE + n + i*n + j  (we store W after x)
        # but X_BASE here is the start of W, not x. We'll pass the W base.
        # R5 <- W_BASE + i*n + j  using shift-and-add (mul by n via add loop is
        # too slow). For demo realism we use Python-side multiply on tiny ints:
        #   R5 = ("LDI", 5, W_BASE)
        #        ("MUL_IMM",) ...
        # but to keep things genuinely multiply-free at the Setun level, we
        # pre-compute the row start address in a separate register that we
        # increment by 1 each j iteration. See ("ADDI", 5, 5, 1) below.
        ("LDX",  3, 5),                      # R3 <- W[address-in-R5]
        ("ADDI", 5, 5, 1),                   # R5 <- R5 + 1  (advance W pointer)
        ("JZ",   3, "skip"),                 # if W[i,j] == 0: skip
        # load x_j: address = X_BASE + j
        ("MOV",  4, 2),                      # R4 <- j  (copied to scratch)
        ("ADDI", 4, 4, X_BASE),              # R4 <- X_BASE + j
        ("LDX",  4, 4),                      # R4 <- x_j  (LDX uses register-indirect)
        ("JN",   3, "subtract"),             # if W < 0: SUB; else fall through to ADD
        ("ADD",  0, 0, 4),                   # R0 += x_j
        ("JMP",  "skip"),
        ("LABEL", "subtract"),
        ("SUB",  0, 0, 4),                   # R0 -= x_j
        ("LABEL", "skip"),
        ("ADDI", 2, 2, 1),                   # j += 1
        ("JMP",  "inner_top"),
        ("LABEL", "inner_done"),
        # write y_i = R0; address = Y_BASE + i
        ("MOV",  4, 1),                      # R4 <- i
        ("ADDI", 4, 4, Y_BASE),              # R4 <- Y_BASE + i
        ("STX",  0, 4),                      # mem[Y_BASE + i] <- R0
        ("ADDI", 1, 1, 1),                   # i += 1
        ("JMP",  "outer_top"),
        ("LABEL", "outer_done"),
        ("HLT",),
    ]


def run_matmul_on_vm(W_list: list[list[int]], x_list: list[int]) -> tuple[list[int], dict]:
    """Run y = W @ x on the SetunVM with a single looped program."""
    m = len(W_list)
    n = len(W_list[0])

    # Memory layout
    X_BASE = 2
    W_BASE = X_BASE + n
    Y_BASE = W_BASE + m * n
    mem_size = Y_BASE + m

    vm = SetunVM(mem_size=mem_size, cycle_limit=10_000_000)
    vm.mem[0] = Trit18.from_int(m, vm.width)
    vm.mem[1] = Trit18.from_int(n, vm.width)
    for j, xv in enumerate(x_list):
        vm.mem[X_BASE + j] = Trit18.from_int(xv, vm.width)
    for i in range(m):
        for j in range(n):
            vm.mem[W_BASE + i * n + j] = Trit18.from_int(W_list[i][j], vm.width)

    program = build_matmul_program(m, n, X_BASE, Y_BASE)

    # Pre-set R5 to W_BASE so the inner-loop loads work.
    # We splice an init instruction at the very start.
    program = [("LDI", 5, W_BASE)] + program

    vm.run(program)
    y_out = [vm.mem[Y_BASE + i].to_int() for i in range(m)]
    return y_out, vm.stats


def main():
    torch.manual_seed(0)
    print("=" * 70)
    print("Cycle-accurate Setun matmul: ONE program walks the whole W and x")
    print("=" * 70)

    # Build a tiny BitLinear, extract weights, run a quantized input through.
    bit = BitLinear(8, 6)
    with torch.no_grad():
        w_q, alpha = ternarize(bit.weight)
        W = w_q.numpy().astype(int).tolist()
        alpha = float(alpha)
    x = torch.randn(8)
    SCALE = 1000
    x_int = (x * SCALE).round().int().tolist()

    print(f"  shape: W is {len(W)}x{len(W[0])}, x has {len(x_int)} ints\n")

    y_int, stats = run_matmul_on_vm(W, x_int)
    print(f"  Setun output (int):  {y_int}")
    print(f"  Reconstructed y:     {[alpha * yi / SCALE for yi in y_int]}")

    # Reference: dense int matmul.
    import numpy as np
    W_np = np.array(W, dtype=np.int64)
    x_np = np.array(x_int, dtype=np.int64)
    y_ref = (W_np @ x_np).tolist()
    print(f"  Reference (W @ x):   {y_ref}")
    assert y_int == y_ref, "Setun matmul disagrees with reference"
    print("\n  -> matches reference exactly.")

    print()
    print("=" * 70)
    print(f"Cycle counts (single looped program):")
    print("=" * 70)
    print(f"  total VM cycles:    {stats['cycles']:>10,}")
    print(f"  ADDs:               {stats['add']:>10,}")
    print(f"  SUBs:               {stats['sub']:>10,}")
    print(f"  MULs:               {stats['mul']:>10,}  (zero — no multiply ops)")
    print(f"  jumps:              {stats['jmp']:>10,}")
    print(f"  loads (LD + LDX):   {stats['ld'] + stats['ldx']:>10,}")
    print(f"  stores (ST + STX):  {stats['st'] + stats['stx']:>10,}")
    print(f"  ternary digit full-adders: {stats['trit_full_adds']:>10,}")

    nonzero = sum(1 for row in W for w in row if w != 0)
    print(f"\n  total ADD+SUB ops:                     {stats['add'] + stats['sub']}")
    print(f"  of which are dot-product accumulations: ~{nonzero} (one per nonzero W)")
    print(f"  the rest are pointer-increment ADDIs (address arithmetic)")
    print(f"\n  -> The matmul accumulation uses zero multiplies.")
    print(f"     Address arithmetic uses only +1 increments, also multiply-free.")
    print(f"     A real Setun-style ALU would have these as separate instructions")
    print(f"     for the address path vs the datapath; both are signed adders.")


if __name__ == "__main__":
    main()
