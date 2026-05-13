"""Synthetic physics-text corpus for char-level LM training.

We generate sentences of the form:
    "pendulum: q=1.5 p=0.0 H=0.39"
    "oscillator: omega=2.0 zeta=0.10 t=3.5 x=-0.18"
    "schrodinger: a=-2.0 b=0.5 E0=-0.86"

each backed by a real numerical solution. This is a "physics LLM" the
right way: the model must learn the joint distribution over parameters
and outcomes, which is non-trivial structure to memorize but small
enough to fit in a tiny char-level transformer.
"""

from __future__ import annotations

import math
import random
from typing import Iterator

import numpy as np
import torch

from .data import damped_oscillator
from .hnn import pendulum_hamiltonian
from .schrodinger import ground_state_energy


def _fmt(x: float, digits: int = 2) -> str:
    return f"{x:+.{digits}f}"[1:] if x >= 0 else f"{x:.{digits}f}"


def _fmt_signed(x: float, digits: int = 2) -> str:
    s = f"{x:.{digits}f}"
    if not s.startswith("-"):
        s = "+" + s
    return s


def _sample_oscillator(rng: random.Random) -> str:
    omega = rng.uniform(0.5, 3.0)
    zeta = rng.uniform(0.05, 0.5)
    t = rng.uniform(0.0, 10.0)
    x = damped_oscillator(torch.tensor([t]), torch.tensor([omega]),
                          torch.tensor([zeta])).item()
    return f"oscillator: omega={omega:.2f} zeta={zeta:.2f} t={t:.2f} x={_fmt_signed(x)}"


def _sample_pendulum(rng: random.Random) -> str:
    q = rng.uniform(-2.5, 2.5)
    p = rng.uniform(-2.0, 2.0)
    H = pendulum_hamiltonian(torch.tensor([q]), torch.tensor([p])).item()
    return f"pendulum: q={_fmt_signed(q)} p={_fmt_signed(p)} H={H:.2f}"


def _sample_schrodinger(rng: random.Random, n_grid: int = 96) -> str:
    a = rng.uniform(-3.0, 1.5)
    b = rng.uniform(0.1, 1.0)
    E0 = ground_state_energy(a, b, n=n_grid)
    return f"schrodinger: a={_fmt_signed(a)} b={b:.2f} E0={_fmt_signed(E0)}"


def _sample_freefall(rng: random.Random) -> str:
    g = rng.uniform(8.0, 12.0)
    t = rng.uniform(0.0, 5.0)
    v0 = rng.uniform(-10.0, 10.0)
    h0 = rng.uniform(0.0, 50.0)
    h = h0 + v0 * t - 0.5 * g * t * t
    return f"freefall: g={g:.2f} t={t:.2f} v0={_fmt_signed(v0)} h0={h0:.2f} h={_fmt_signed(h)}"


def _sample_ohm(rng: random.Random) -> str:
    V = rng.uniform(0.5, 24.0)
    R = rng.uniform(1.0, 1000.0)
    I = V / R
    P = V * I
    return f"ohm: V={V:.2f} R={R:.2f} I={I:.4f} P={P:.4f}"


def _sample_relativistic(rng: random.Random) -> str:
    # E^2 = (pc)^2 + (m c^2)^2 ; with c=1: E = sqrt(p^2 + m^2)
    m = rng.uniform(0.1, 10.0)
    p = rng.uniform(-10.0, 10.0)
    E = math.sqrt(p * p + m * m)
    return f"relativistic: m={m:.2f} p={_fmt_signed(p)} E={E:.4f}"


def _sample_planet(rng: random.Random) -> str:
    # Kepler's third law: T^2 = a^3 (G M = 1 implicit units)
    a = rng.uniform(0.4, 30.0)
    T = a ** 1.5
    return f"kepler: a={a:.2f} T={T:.4f}"


def _sample_blackbody(rng: random.Random) -> str:
    # Wien displacement: lambda_max * T = b, with b = 2898 um K
    T = rng.uniform(1000.0, 10000.0)
    lam = 2898.0 / T
    return f"wien: T={T:.0f} lambda_max={lam:.4f}"


def _sample_doppler(rng: random.Random) -> str:
    # Doppler shift (non-relativistic): f_obs = f_src * (1 - v/c)
    f_src = rng.uniform(100.0, 1000.0)
    v_over_c = rng.uniform(-0.1, 0.1)
    f_obs = f_src * (1.0 - v_over_c)
    return f"doppler: f_src={f_src:.2f} v_c={_fmt_signed(v_over_c)} f_obs={f_obs:.2f}"


_DEFAULT_SAMPLERS = [
    _sample_oscillator,
    _sample_pendulum,
    _sample_schrodinger,
    _sample_freefall,
    _sample_ohm,
    _sample_relativistic,
    _sample_planet,
    _sample_blackbody,
    _sample_doppler,
]


def build_corpus(n_lines: int = 4000, seed: int = 0,
                 samplers: list | None = None) -> str:
    rng = random.Random(seed)
    if samplers is None:
        samplers = _DEFAULT_SAMPLERS
    lines = []
    for _ in range(n_lines):
        fn = rng.choice(samplers)
        lines.append(fn(rng))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Tokenization (char-level)
# ---------------------------------------------------------------------------

class CharVocab:
    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.size = len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


def make_batches(text: str, vocab: CharVocab, block_size: int, batch_size: int,
                 seed: int = 0) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    data = torch.tensor(vocab.encode(text), dtype=torch.long)
    n = data.shape[0]
    g = torch.Generator().manual_seed(seed)
    while True:
        ix = torch.randint(0, n - block_size - 1, (batch_size,), generator=g)
        x = torch.stack([data[i:i + block_size] for i in ix])
        y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
        yield x, y
