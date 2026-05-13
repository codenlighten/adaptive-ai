"""Tiny Setun — a balanced-ternary ALU + minimal VM.

Named after the 1958 Soviet ternary computer built at Moscow State University
under Brusentsov and Sobolev. The original Setun used 18-trit words; this
module makes the width configurable but defaults to 18.

What's here:
  - Trit18: a balanced-ternary signed integer (default 18 trits = ~ +/-193M)
  - Arithmetic on Trit18: add, sub, neg, mul, compare, shift
       — implemented without using Python's built-in `*` on the digits
       (only +, -, and table lookups), so it could in principle be
       transliterated into a real ternary logic gate netlist
  - SetunVM: a tiny register machine with balanced-ternary opcodes
  - A worked example: matrix-vector multiply for a small ternary-weight
    BitLinear layer, executed entirely as a sequence of Setun instructions

The point isn't speed (it's Python — orders of magnitude slower than NumPy);
the point is to show that the entire stack — storage (packed trits),
arithmetic (signed adds), inference (no multiplies), all the way down
to instruction-level computation — can be made out of nothing but trits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

# ---------------------------------------------------------------------------
# Full-adder for balanced ternary digits
#
# Each input is in {-1, 0, +1}. Their sum is in {-2, -1, 0, +1, +2}. To stay
# in balanced ternary we represent: -2 = -1*3 + 1 (carry -1, sum digit +1),
# +2 = +1*3 + -1 (carry +1, sum digit -1).
#
# This table is the heart of a balanced-ternary ALU. It's the equivalent of
# the binary half/full adder, and could be realized in three-state CMOS,
# carbon nanotube ternary logic, or Josephson junctions.
# ---------------------------------------------------------------------------

# (a + b + cin) -> (sum_digit, cout); all in {-1, 0, +1}
def _full_add_trit(a: int, b: int, cin: int) -> tuple[int, int]:
    s = a + b + cin
    if s == -2:
        return 1, -1
    if s == -1:
        return -1, 0
    if s == 0:
        return 0, 0
    if s == 1:
        return 1, 0
    if s == 2:
        return -1, 1
    if s == 3:                          # 1+1+1
        return 0, 1
    if s == -3:                         # -1+-1+-1
        return 0, -1
    raise ValueError(f"unreachable sum {s}")


# ---------------------------------------------------------------------------
# Trit18: a balanced-ternary signed integer.
# Trits are stored little-endian (index 0 = least significant trit).
# ---------------------------------------------------------------------------

@dataclass
class Trit18:
    """Balanced-ternary integer with `width` trits (default 18 — Setun word size).

    Negative values are represented natively: just flip the sign of every trit.
    No two's-complement gymnastics needed. This is one of the elegances of
    balanced ternary.
    """

    digits: list[int]
    width: int = 18

    def __post_init__(self):
        assert len(self.digits) == self.width
        for d in self.digits:
            assert d in (-1, 0, 1), f"non-ternary digit {d}"

    @classmethod
    def from_int(cls, n: int, width: int = 18) -> "Trit18":
        digits: list[int] = []
        x = n
        for _ in range(width):
            r = x % 3
            x = x // 3
            # Convert to balanced: 0,1 unchanged; 2 -> -1 and bump x.
            if r == 2:
                r = -1
                x += 1
            digits.append(r)
        if x != 0:
            raise OverflowError(f"value {n} does not fit in {width} balanced trits")
        return cls(digits, width)

    def to_int(self) -> int:
        n = 0
        pow3 = 1
        for d in self.digits:
            n += d * pow3
            pow3 *= 3
        return n

    def __repr__(self) -> str:
        # Print most-significant trit first, with conventional notation:
        # T for -1, 0, 1.
        out = []
        for d in reversed(self.digits):
            out.append("T" if d == -1 else str(d))
        return f"Trit18({''.join(out)}={self.to_int()})"


# ---------------------------------------------------------------------------
# Arithmetic — defined only in terms of _full_add_trit and trit negation.
# These are the primitives a Setun ALU would actually execute.
# ---------------------------------------------------------------------------

def neg(a: Trit18) -> Trit18:
    return Trit18([-d for d in a.digits], a.width)


def add(a: Trit18, b: Trit18) -> Trit18:
    assert a.width == b.width
    out = [0] * a.width
    carry = 0
    for i in range(a.width):
        s, carry = _full_add_trit(a.digits[i], b.digits[i], carry)
        out[i] = s
    if carry != 0:
        raise OverflowError("balanced-ternary add overflow")
    return Trit18(out, a.width)


def sub(a: Trit18, b: Trit18) -> Trit18:
    return add(a, neg(b))


def compare(a: Trit18, b: Trit18) -> int:
    """Returns -1 if a<b, 0 if a==b, +1 if a>b.

    Notice the elegance: this is just the sign of the most-significant
    nonzero trit of (a - b). Native, no special-cases.
    """
    d = sub(a, b)
    for i in reversed(range(d.width)):
        if d.digits[i] != 0:
            return d.digits[i]
    return 0


def shift_left(a: Trit18, k: int) -> Trit18:
    """Multiply by 3^k (positional shift). Equivalent to << in binary."""
    if k == 0:
        return Trit18(a.digits[:], a.width)
    if k < 0:
        return shift_right(a, -k)
    out = [0] * k + a.digits[:-k]
    return Trit18(out, a.width)


def shift_right(a: Trit18, k: int) -> Trit18:
    if k == 0:
        return Trit18(a.digits[:], a.width)
    if k < 0:
        return shift_left(a, -k)
    out = a.digits[k:] + [0] * k
    return Trit18(out, a.width)


def mul(a: Trit18, b: Trit18) -> Trit18:
    """Multiplication via shift-and-add.

    For each trit b_i of b at position i:
      if b_i == +1: accumulator += a << i
      if b_i == -1: accumulator -= a << i
      if b_i ==  0: skip
    No scalar-on-scalar multiplications are used anywhere — only adds, subs,
    and positional shifts. This is exactly how a Setun multiplier worked.
    """
    acc = Trit18.from_int(0, a.width)
    for i, bi in enumerate(b.digits):
        if bi == 0:
            continue
        shifted = shift_left(a, i)
        if bi == 1:
            acc = add(acc, shifted)
        else:  # bi == -1
            acc = sub(acc, shifted)
    return acc


# Quick sanity-check primitive: dot product of a ternary-weight row by an
# integer-valued input vector — using ONLY add/sub/skip on Trit18s.
def ternary_dot(weights: list[int], inputs: list[Trit18]) -> Trit18:
    """Compute sum_i w_i * x_i where w_i in {-1, 0, +1}.

    Uses only add and sub on Trit18 — exactly the multiply-free matmul,
    but expressed at the Setun instruction level.
    """
    assert len(weights) == len(inputs)
    width = inputs[0].width
    acc = Trit18.from_int(0, width)
    for w, x in zip(weights, inputs):
        if w == 1:
            acc = add(acc, x)
        elif w == -1:
            acc = sub(acc, x)
        # else: skip
    return acc


# ---------------------------------------------------------------------------
# A minimal Setun-style virtual machine.
#
# Registers R0..R7 each hold a Trit18. Memory is a list of Trit18s. Opcodes
# are themselves ternary-named for fun. A program is a list of (op, args).
# ---------------------------------------------------------------------------

class SetunVM:
    """Register VM with balanced-ternary arithmetic. 8 regs, configurable memory.

    Supports control flow (JMP, JZ, JNZ, JS) so you can write looped programs
    that walk over memory — the way a real Setun program would be structured.
    """

    OPS: dict[str, Callable] = {}

    def __init__(self, mem_size: int = 64, width: int = 18,
                 cycle_limit: int = 1_000_000):
        self.width = width
        self.regs: list[Trit18] = [Trit18.from_int(0, width) for _ in range(8)]
        self.mem: list[Trit18] = [Trit18.from_int(0, width) for _ in range(mem_size)]
        self.pc: int = 0
        self.halted: bool = False
        self.cycle_limit = cycle_limit
        # Stats — how many of each kind of operation were performed
        self.stats = {"add": 0, "sub": 0, "neg": 0, "mul": 0,
                      "ld": 0, "st": 0, "mov": 0, "cmp": 0, "jmp": 0,
                      "ldx": 0, "stx": 0, "trit_full_adds": 0,
                      "cycles": 0}
        # `labels` is a dict of label_name -> instruction index for assembly-style
        self.labels: dict[str, int] = {}

    def _add(self, a: Trit18, b: Trit18) -> Trit18:
        self.stats["trit_full_adds"] += a.width
        return add(a, b)

    def _resolve(self, target):
        """Resolve a jump target — int or label name."""
        if isinstance(target, str):
            return self.labels[target]
        return int(target)

    def step(self, instr) -> bool:
        """Execute one instruction. Returns True if PC was set explicitly (jump)."""
        op = instr[0]
        if op == "LDI":          # LDI rd, immediate
            _, rd, imm = instr
            self.regs[rd] = Trit18.from_int(imm, self.width)
        elif op == "LD":         # LD rd, addr_const
            _, rd, addr = instr
            self.regs[rd] = self.mem[addr]
            self.stats["ld"] += 1
        elif op == "LDX":        # LDX rd, ra  -- rd <- mem[ra.to_int()]
            _, rd, ra = instr
            self.regs[rd] = self.mem[self.regs[ra].to_int()]
            self.stats["ldx"] += 1
        elif op == "ST":         # ST rs, addr_const
            _, rs, addr = instr
            self.mem[addr] = self.regs[rs]
            self.stats["st"] += 1
        elif op == "STX":        # STX rs, ra  -- mem[ra.to_int()] <- rs
            _, rs, ra = instr
            self.mem[self.regs[ra].to_int()] = self.regs[rs]
            self.stats["stx"] += 1
        elif op == "MOV":        # MOV rd, rs
            _, rd, rs = instr
            self.regs[rd] = self.regs[rs]
            self.stats["mov"] += 1
        elif op == "ADD":        # ADD rd, ra, rb
            _, rd, ra, rb = instr
            self.regs[rd] = self._add(self.regs[ra], self.regs[rb])
            self.stats["add"] += 1
        elif op == "ADDI":       # ADDI rd, ra, imm
            _, rd, ra, imm = instr
            self.regs[rd] = self._add(self.regs[ra], Trit18.from_int(imm, self.width))
            self.stats["add"] += 1
        elif op == "SUB":        # SUB rd, ra, rb
            _, rd, ra, rb = instr
            self.regs[rd] = self._add(self.regs[ra], neg(self.regs[rb]))
            self.stats["sub"] += 1
        elif op == "NEG":        # NEG rd, rs
            _, rd, rs = instr
            self.regs[rd] = neg(self.regs[rs])
            self.stats["neg"] += 1
        elif op == "MUL":        # MUL rd, ra, rb
            _, rd, ra, rb = instr
            self.regs[rd] = mul(self.regs[ra], self.regs[rb])
            self.stats["mul"] += 1
        elif op == "CMP":        # CMP rd, ra, rb  -- rd <- sign(ra - rb)
            _, rd, ra, rb = instr
            s = compare(self.regs[ra], self.regs[rb])
            self.regs[rd] = Trit18.from_int(s, self.width)
            self.stats["cmp"] += 1
        elif op == "JMP":        # JMP target
            self.pc = self._resolve(instr[1])
            self.stats["jmp"] += 1
            return True
        elif op == "JZ":         # JZ rs, target — jump if rs == 0
            _, rs, target = instr
            self.stats["jmp"] += 1
            if self.regs[rs].to_int() == 0:
                self.pc = self._resolve(target)
                return True
        elif op == "JNZ":        # JNZ rs, target — jump if rs != 0
            _, rs, target = instr
            self.stats["jmp"] += 1
            if self.regs[rs].to_int() != 0:
                self.pc = self._resolve(target)
                return True
        elif op == "JP":         # JP rs, target — jump if rs > 0
            _, rs, target = instr
            self.stats["jmp"] += 1
            if self.regs[rs].to_int() > 0:
                self.pc = self._resolve(target)
                return True
        elif op == "JN":         # JN rs, target — jump if rs < 0
            _, rs, target = instr
            self.stats["jmp"] += 1
            if self.regs[rs].to_int() < 0:
                self.pc = self._resolve(target)
                return True
        elif op == "LABEL":      # pseudo-op; ignored at runtime
            pass
        elif op == "HLT":
            self.halted = True
        else:
            raise ValueError(f"unknown op {op}")
        return False

    def load_program(self, program: list[tuple]) -> None:
        """Resolve labels in the program."""
        self.labels.clear()
        for i, instr in enumerate(program):
            if instr[0] == "LABEL":
                self.labels[instr[1]] = i

    def run(self, program: list[tuple]) -> None:
        """Run a program with control flow until HLT or cycle limit."""
        self.load_program(program)
        self.pc = 0
        while not self.halted and self.pc < len(program):
            if self.stats["cycles"] >= self.cycle_limit:
                raise RuntimeError(f"cycle limit {self.cycle_limit} exceeded")
            instr = program[self.pc]
            jumped = self.step(instr)
            self.stats["cycles"] += 1
            if not jumped:
                self.pc += 1


def matmul_as_setun_program(W: list[list[int]], x_vals: list[int]) -> tuple[list[tuple], SetunVM]:
    """Generate a Setun program that computes y = W @ x for ternary W and integer x.

    Layout:
      mem[0..n-1]   = x_vals (loaded as Trit18)
      mem[n..n+m-1] = y outputs (written by program)

    Strategy per output row i: zero R0 (accumulator), then for each j with
    W[i][j] != 0, load x_j into R1 and either ADD or SUB R0 += R1.
    Store R0 to mem[n+i]. No MULs used — this is the multiply-free matmul.
    """
    m = len(W)         # output dim
    n = len(W[0])      # input dim
    program: list[tuple] = []

    vm = SetunVM(mem_size=m + n)
    # Pre-load inputs into memory.
    for j, xv in enumerate(x_vals):
        vm.mem[j] = Trit18.from_int(xv, vm.width)

    for i in range(m):
        program.append(("LDI", 0, 0))                # R0 <- 0 (accumulator)
        for j in range(n):
            w = W[i][j]
            if w == 0:
                continue
            program.append(("LD", 1, j))             # R1 <- x_j
            if w == 1:
                program.append(("ADD", 0, 0, 1))     # R0 <- R0 + R1
            else:
                program.append(("SUB", 0, 0, 1))     # R0 <- R0 - R1
        program.append(("ST", 0, n + i))             # mem[n+i] <- R0
    program.append(("HLT",))
    return program, vm
