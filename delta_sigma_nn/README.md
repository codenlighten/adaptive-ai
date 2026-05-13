# delta-sigma-nn

**Multiply-free neural networks with delta-sigma weights and anytime inference.**

Each weight is encoded as a length-T trit stream whose time-average reconstructs
the underlying real value. Every cycle of the matmul uses only signed additions,
eliminating multiplications entirely; the precision–compute tradeoff is
controlled at *inference time* by truncating the stream.

## Installation

```bash
pip install delta-sigma-nn
```

## Quick start

```python
import torch
from delta_sigma_nn import DeltaSigmaMLP

# Build a 5-layer MLP with delta-sigma weights at T=8 time steps
model = DeltaSigmaMLP(in_dim=3, hidden_dim=128, out_dim=1, depth=5, T=8)

# Train with normal PyTorch loops
opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
for _ in range(100):
    opt.zero_grad()
    ((model(X) - y) ** 2).mean().backward()
    opt.step()

# Anytime inference: choose accuracy/compute tradeoff at runtime
y_fast, k = model.anytime_inference(x, stop_eps=0.1)    # uses ~3 of 8 steps
y_accurate, k = model.anytime_inference(x, stop_eps=0.01)  # uses ~8 of 8
```

## Public API

| symbol | description |
|---|---|
| `DeltaSigmaLinear` | drop-in replacement for `nn.Linear` with DS weights |
| `DeltaSigmaMLP` | convenience wrapper for an MLP architecture |
| `DSigmaCharLM` | causal char-level LM with DS weights everywhere |
| `save_dsigma_mlp` / `load_dsigma_arrays` | packed-stream serialization |
| `dsigma_inference` | pure-NumPy inference engine with anytime-mode `k` |
| `encode_delta_sigma_ternary` / `encode_delta_sigma_order2` | modulators |

## Results

Validated on three tasks (5-layer 128-d MLPs, identical training recipe):

| Task | BitMLP (ternary) | fp32 | DSigma best |
|---|---:|---:|---:|
| Damped oscillator (MSE) | 1.0e-4 | 2.2e-5 | **6.0e-5** (T=8) |
| Schrödinger E₀ (MSE) | 1.6e-3 | 1.1e-4 | **5.2e-4** (T=16) |
| Digits 10-way classification (acc) | 97.78% | 98.06% | **98.06%** (T=16, ties fp32) |

Anytime inference (T=32 trained model, per-example k truncation):

| stop_eps | avg k of 32 | compute reduction |
|---:|---:|---:|
| 0.5 | 2.0 | 16× |
| 0.1 | 2.5 | 13× |
| 0.05 | 3.1 | 10× |
| 0.01 | 7.9 | 4× |

## Hardware

A synthesizable Verilog `dsigma_pe` is included in the source repo. On a
Lattice iCE40 HX1K (a $5 hobbyist FPGA), one PE uses **206 logic cells (16%
of fabric)** and runs at **75 MHz max clock**.

For comparison, the plain ternary PE (without delta-sigma encoding) uses
283 LCs at 16.83 MHz on the same fabric. The shift-register architecture
of delta-sigma allows the synthesizer to share resources and shorten the
critical path.

## Citing / details

For the full technical writeup, including the mechanism derivation,
related work, and limitations, see [DELTA_SIGMA_WEIGHTS.md](https://github.com/example/delta-sigma-nn/blob/main/DELTA_SIGMA_WEIGHTS.md).
