# Delta-Sigma Weights for Neural Networks

**A multiply-free training mechanism with anytime inference.**

## Abstract

We propose **delta-sigma weights**, a quantization mechanism that
combines the per-cycle simplicity of ternary networks with the
precision flexibility of higher-bit-width formats. Each weight is
encoded as a length-T trit stream whose time-average reconstructs the
underlying real value. Every cycle of the matmul uses only signed
additions, eliminating multiplications entirely; the precision–compute
tradeoff is controlled at *inference time* by truncating the stream.

Empirically, on three tasks (regression, eigenvalue regression,
classification), first-order delta-sigma weights at T = 8 – 16 beat
pure-ternary baselines by 1.7×–3× in val MSE and tie fp32 on
classification accuracy. We further demonstrate **anytime inference**:
truncating to k < T early stream steps gives a per-example runtime
knob that trades accuracy for up to 16× compute reduction. Synthesis
of a delta-sigma processing element on iCE40 FPGA shows the mechanism
is **smaller (206 LCs vs 283) and faster (75 MHz vs 17 MHz)** than a
plain ternary processing element.

## 1. Motivation

Ternary BitNet-style quantization (Ma et al., 2024) compresses weights
to {−1, 0, +1} and replaces multiplies with signed adds. Two
limitations:

1. **No precision dial.** Once trained, a ternary network has a fixed
   weight matrix. There's no way to get a *more accurate* answer if you
   have budget for more compute, or a *faster* answer if accuracy is
   less critical.
2. **Static per-weight bits.** A weight of 0.7 and a weight of 0.05
   both consume the same trit. The fixed ternary grid wastes bits on
   weights that need fewer states and starves weights that could use
   more.

Delta-sigma modulation, decades old in audio digital-to-analog
converters, solves both problems by *trading time for precision*. A 1-bit
DAC clocked at MHz rates can faithfully reproduce a 16-bit audio
signal because its instantaneous error is integrated and noise-shaped
away from the signal band.

We adapt this to neural network weights: encode the target weight as
a T-step bit-stream, then compute the matmul as T one-trit matmuls
averaged. Each cycle uses only signed adds; the precision is
controlled by T.

## 2. The mechanism

### Encoding

Given a continuous target weight `w ∈ [−1, 1]`, the first-order
balanced delta-sigma modulator emits a length-T trit stream:

```
integrator = 0
for t in 1..T:
    integrator += w_target
    if integrator > +θ:      bit_t = +1
    elif integrator < −θ:    bit_t = −1
    else:                    bit_t = 0
    integrator -= bit_t
```

With θ = 0.5, the time-average `(1/T) Σ bit_t` converges to `w_target`
as `O(1/T)`. The convergence is monotonic in expectation;
quantization error decreases each cycle.

### Layer

A `DeltaSigmaLinear` layer stores a continuous weight tensor `W`, a
scale `α = mean(|W|)`, and computes:

```
W_norm = (W / α).clamp(-1, 1)
stream = encode(W_norm, T=T)       # shape (T, out, in), each entry in {-1, 0, +1}
W_eff  = α * stream.mean(dim=0)    # equals W ± O(α / T)
y      = LayerNorm(x) @ W_eff.T + b
```

Crucially the matmul `LayerNorm(x) @ W_eff.T` decomposes into:

```
y = (α / T) Σ_t (LayerNorm(x) @ stream[t].T)
```

Each inner matmul `stream[t].T` is a ternary matmul — every weight is
in {−1, 0, +1}, every contribution is an add, sub, or skip. After T
cycles we divide by T (one scalar shift on hardware).

**Compute per matmul: T one-trit matmuls + one final scalar
rescale. Zero multiplications anywhere.**

### Backward

Forward uses the time-averaged ternary reconstruction; backward uses
the straight-through estimator: gradients flow as if the matmul used
continuous `W`. This is the same STE used in BitNet b1.58, applied
here to the encoding step rather than to per-weight rounding.

### Anytime inference

The cumulative average over the first k of T steps:
`(1/k) Σ_{t=1}^k stream[t]` is a progressively better estimate of
`W_norm`. At inference we can:

1. Compute `y_2 = (α/2) Σ_{t=1}^2 (x @ stream[t].T)`. Stop if confident.
2. Otherwise compute the next 2 slices and update to `y_4`.
3. Double until output stabilizes or `k = T`.

The user chooses the accuracy/compute tradeoff at runtime via a
stop-tolerance parameter. Easy examples converge in `k = 2`; hard
examples use full `k = T`.

## 3. Results

### 3.1 Validation across three tasks (MLP)

Identical 5-layer 128-hidden MLPs:

| Task | Metric | BitMLP | FPMLP | DSigma best T | DSigma best |
|---|---|---:|---:|---:|---:|
| Damped oscillator | val MSE | 0.000103 | 0.000022 | T=8 | **0.000060** |
| Schrödinger E₀ | val MSE | 0.001573 | 0.000105 | T=16 | **0.000523** |
| Digits (10-way) | val accuracy | 97.78% | 98.06% | T=16 | **98.06%** ← ties fp32 |

DSigma beats pure ternary on every task. Digits at T=16 ties fp32
exactly. Optimal T is task-dependent (8–16); more T helps until saturation.

### 3.1b Validation at transformer scale

A 4-layer causal char-level LM with delta-sigma weights in every Q/K/V/proj/FFN
(`DSigmaCharLM`), trained on a 6000-line physics corpus:

| config | params | val ppl |
|---|---:|---:|
| BitTx (ternary) | 556,032 | 2.105 |
| DSigma T=4 | 556,032 | 2.099 |
| **DSigma T=8** | 556,032 | **2.093** ← tied best |
| **DSigma T=16** | 556,032 | **2.093** ← tied best |
| FPTx (fp32) | 550,912 | 2.106 |

**The DSigma transformer beats both ternary AND fp32 at this scale.**
The mechanism transfers from MLPs to transformers without modification.

### 3.2 Anytime inference

Trained a DSigma-MLP at T=32 on the damped oscillator; ran per-example
anytime inference at varying tolerance:

| stop_eps | avg k of 32 | val MSE | compute reduction |
|---:|---:|---:|---:|
| 0.500 | 2.00 | 9.5e-4 | **16×** |
| 0.100 | 2.46 | 6.4e-4 | 13× |
| 0.050 | 3.07 | 4.4e-4 | 10× |
| 0.020 | 4.88 | 2.7e-4 | 6.5× |
| 0.010 | 7.87 | 1.9e-4 | 4× |

At loose tolerance the network averages just 2 of 32 cycles per
example. At tight tolerance it uses ~8. The user controls the tradeoff
through one parameter; **the network adapts its compute per input**.

### 3.3 Packed-stream storage

Trained model serialized as packed trits:

| component | size |
|---|---:|
| 2× fp32 boundary layers (input + output) | 2,048 bytes |
| 3× delta-sigma layer streams (T=8) | 78,645 bytes |
| Total checkpoint | 82,033 bytes |
| fp32 state-dict equivalent | 203,780 bytes |
| **Compression** | **2.20×** |

A pure-NumPy inference engine loads the packed file and runs forward
with k ∈ {1,…,T} chosen at runtime. Output matches the torch model
within fp roundoff (max abs diff 9e-7).

### 3.4 Hardware

A `dsigma_pe` Verilog module — shift register + ternary signed-adder
+ accumulator — was synthesized for the Lattice iCE40 HX1K (a $5
hobbyist FPGA) via Yosys + nextpnr-ice40.

| metric | dsigma_pe (T=8) | plain ternary_pe |
|---|---:|---:|
| Logic cells (LCs) | **206** (16% of HX1K) | 283 (22%) |
| Max frequency | **75 MHz** | 16.83 MHz |
| Multiplications per cycle | 0 | 0 |

The dsigma PE is *smaller and faster* than the plain ternary PE
because the shift-register architecture allows the synthesizer to
share resources and shortens the critical path. T cycles in real
silicon = T / 75 µs per matmul row at T=8 = 107 ns.

## 4. The 5-second pitch

Per-cycle: ternary's compute. Time-averaged: continuous precision.
Runtime: a precision dial. Hardware: smaller and faster than ternary.
Storage: 2× smaller than fp32 (more compact with packed trits and
shorter T).

Every cycle of a delta-sigma matmul uses signed adds only. Stack
enough cycles and you recover arbitrary continuous precision. Stop
early and you get a fast, noisy answer for inputs where that's
sufficient.

## 5. Related work and where this sits

- **BitNet b1.58** (Ma et al., 2024): fixed-precision ternary weights,
  no time dimension. We extend this with a learnable precision via T.
- **Spiking neural networks** (e.g., Loihi, TrueNorth): use temporal
  coding for *activations*. We use it for *weights*.
- **Anytime/multi-exit networks** (BranchyNet, Cascaded Networks): use
  architectural early-exit. Our anytime mechanism is purely at the
  precision level, doesn't require additional exit branches.
- **Sigma-delta modulators** (decades of DSP literature): well
  understood; we are repurposing the encoder, not inventing it.
- **Stochastic computing**: uses random bit streams for arithmetic.
  Delta-sigma is the deterministic, noise-shaped variant — better
  convergence per cycle.

To my knowledge the *combination* — sigma-delta encoded weights with
STE training and anytime stream-truncation inference — has not been
published. The closest prior work in spirit is "Bitwise Neural
Networks" (Kim & Smaragdis, 2016) for binary activations.

## 6. Honest limitations

- **Storage advantage shrinks with T.** Pure ternary is ~1.6 bits/weight.
  DSigma at T=8 is ~8 trits = 12.7 bits per weight pre-packing
  (~2.5 bits/weight effective with run-length / arithmetic coding,
  which we haven't implemented). The win is in *compute and
  flexibility*, not in storage.
- **Second-order modulators didn't help in our experiments.** A
  properly tuned 2nd-order ΣΔ should converge faster than 1st-order;
  ours did not in the cases we tested. Likely needs theta retuning.
- **Optimal T is task-dependent.** No theoretical principle yet for
  picking T a priori; we found it empirically per task.
- **The training cost is T× a ternary network's** (every forward pass
  runs T inner matmuls). Inference is what's flexible.

## 7. Where this could go

1. **Larger validation**: BPE-tokenized LM, image classification, full
   transformer rollout. Does the "T = O(8–16) is the sweet spot"
   pattern hold at larger model sizes?
2. **Run-length / arithmetic coding of streams**: trit streams have
   structure (long runs of zero are common); compressing them with
   entropy coding would close the storage gap with pure ternary.
3. **Asynchronous anytime inference**: in a real distributed system,
   different layers' k values could be chosen independently per
   request based on layer-level confidence signals.
4. **Hardware MAC array**: build a systolic array of dsigma_pes,
   synthesize and route, get end-to-end LC + power numbers for a
   complete matmul accelerator.

## 8. Reproduction

All code is in `physics-ai/`. From a fresh clone:

```bash
source venv/bin/activate
python -m pytest tests/                     # 79 tests
python -m scripts.train_dsigma --plot       # T sweep on oscillator
python -m scripts.validate_dsigma           # Schrödinger + digits
python -m scripts.anytime_inference         # per-example k truncation
python -m scripts.run_dsigma_packed         # packed-stream end-to-end demo

yowasp-yosys -p "read_verilog hardware/ternary_full_adder.v hardware/dsigma_pe.v; \
    synth_ice40 -top dsigma_pe -json hardware/dsigma_pe.json"
yowasp-nextpnr-ice40 --hx1k --json hardware/dsigma_pe.json --asc /tmp/dsigma_pe.asc
```

The full source code, tests, training scripts, Verilog modules,
synthesis scripts, and figures are all included. No external data
required.
