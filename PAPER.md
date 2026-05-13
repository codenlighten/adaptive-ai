# Delta-Sigma Weights: Multiply-Free Neural Networks with Anytime Inference

**Gregory J. Ward**, **Bryan W. Daugherty**, **Shawn M. Ryan**
*SmartLedger.Technology*
Corresponding author: `codenlighten1@gmail.com`

## Abstract

We introduce *delta-sigma weights*, a quantization mechanism for neural
networks in which each weight is encoded as a length-T trit stream
whose time-average reconstructs the underlying real value. Every cycle
of the resulting matrix multiplication uses only signed additions on
trits in {−1, 0, +1}; no floating-point multiplications occur anywhere
on the data path. The precision–compute tradeoff is controlled by T at
training time, and by a per-example truncation k ≤ T at inference time.
Across four tasks — damped oscillator regression, 1-D Schrödinger
ground-state energy regression, sklearn-digits 10-way classification,
and a 556k-parameter causal char-level transformer — delta-sigma
weights match or exceed pure-ternary baselines and, at transformer
scale, exceed fp32 perplexity. We further demonstrate **anytime
inference**: per-example truncation of the stream yields 4×–16×
compute reduction with graceful accuracy degradation. A Verilog
processing element synthesized for the Lattice iCE40 FPGA uses
**fewer logic cells and a higher clock frequency** than a plain
ternary processing element (206 LCs at 75 MHz vs. 283 LCs at 17 MHz).
The full mechanism is implemented in an open-source Python package
`delta_sigma_nn`, with a pure-NumPy inference engine over packed trit
streams that matches the PyTorch model within fp roundoff.

## 1. Introduction

Quantized neural networks reduce the cost of inference by replacing
floating-point operations with integer or sub-integer arithmetic. The
most aggressive practical scheme to date is **BitNet b1.58** (Ma et
al., 2024), which constrains weights to the balanced-ternary set
{−1, 0, +1} and demonstrates parity with fp16 at 3B–7B parameters
while reducing inference energy by ~4× and DRAM bandwidth by ~10×.

Ternary networks have two structural limitations we address here:

1. **Rigid precision.** Once trained, a ternary network has a fixed
   weight matrix; there is no mechanism to obtain a *more accurate*
   output by spending additional compute, nor a *faster* output by
   accepting reduced accuracy.
2. **Mismatched bit budgets.** A weight of magnitude 0.7 and a weight
   of magnitude 0.05 consume the same trit. Weights that need more
   precision are starved; weights that need less are wasted.

Both limitations suggest a missing degree of freedom: a precision
*dial* that operates at the per-weight level, controllable at
inference time.

We propose **delta-sigma weights**, a quantization scheme borrowed
from oversampled digital-to-analog conversion in DSP. Each weight `w`
is encoded as a length-T trit stream
`b₁, b₂, …, b_T ∈ {−1, 0, +1}`
whose time-average converges to `w`:

```
mean(b_1, …, b_T) → w   as T → ∞,   with error O(1/T) for first-order
```

The matrix multiplication `y = W · x` is then computed as a time-loop
of T one-trit matrix multiplications, averaged and rescaled by a
per-layer scalar α. **No cycle of this computation uses a
floating-point multiplication.** Increasing T monotonically improves
the effective precision of `W`; at inference, the running cumulative
average over the first k of T steps provides progressively-better
estimates, enabling early-exit per example.

### Contributions

We claim three contributions:

1. **A novel quantization mechanism** that combines the per-cycle
   simplicity of ternary networks with continuous precision via
   delta-sigma modulation, applied for the first time (to our
   knowledge) to neural-network weights as a training-time and
   inference-time mechanism.

2. **Empirical validation** on four tasks (three MLP regressions /
   classifications, one causal transformer LM). Delta-sigma weights
   match or exceed pure ternary on all four tasks and **exceed fp32
   perplexity at 556k parameters on the transformer task**.

3. **A hardware implementation**: a synthesizable Verilog processing
   element placed and routed on a $5 Lattice iCE40 FPGA, occupying
   **fewer logic cells and running at higher clock frequency** than a
   plain ternary processing element on the same fabric. Combined with
   a packed-trit storage format and a pure-NumPy reference inference
   engine, the entire chain — gradient descent through silicon — is
   end-to-end reproducible.

## 2. Background

### 2.1 BitNet b1.58 and ternary weight quantization

BitNet b1.58 quantizes each weight to {−1, 0, +1} times a per-tensor
scale α, with a threshold rule `θ = 0.75 · mean(|W|)` below which
weights snap to zero. Activations are kept in higher precision
(typically int8 or fp16). Training uses the straight-through estimator
(Bengio et al., 2013): forward propagation uses the quantized weights,
backward propagation treats the quantizer as an identity.

The resulting matmul `y = αW' · x` with `W' ∈ {−1, 0, +1}^{m × n}` has
the property that each weight contributes to the dot product as either
`+x_j`, `−x_j`, or zero — no multiplication needed. On custom silicon
where multipliers dominate area and energy, ternary networks therefore
deliver substantial gains.

### 2.2 Sigma-delta modulation

Sigma-delta (ΣΔ) modulators are oversampled digital-to-analog
converters that emit low-bit-width samples whose time-average
reconstructs a higher-bit-width input. The first-order balanced
modulator is:

```
integrator ← 0
for t in 1..T:
    integrator ← integrator + x_target
    b_t ← Q(integrator)        # quantize to {−1, 0, +1} (or {−1, +1})
    integrator ← integrator − b_t
emit b_1, …, b_T
```

The cumulative average `(1/T) Σ b_t` converges to `x_target` with
error `O(1/T)` for first-order modulators. Higher-order modulators
*shape* the quantization noise to high frequencies, accelerating
convergence at the cost of additional integrators and stability
concerns.

ΣΔ modulators are textbook content in DSP (Schreier and Temes, 2004)
but, to our knowledge, have not been applied as a training-time
mechanism for neural-network weights.

### 2.3 Anytime and adaptive-precision inference

Anytime inference — producing usable outputs after partial computation
— has been explored architecturally (BranchyNet, Teerapittayanon et
al., 2016; multi-exit networks; cascaded classifiers) and via mixed
precision schemes (HAQ, Wang et al., 2019; once-for-all, Cai et al.,
2020). Our mechanism differs in that the anytime dimension is *built
into the weight representation*: every layer has the same anytime
truncation knob `k`, controlled at runtime per-example by a single
parameter, with no architectural duplication.

## 3. Method: Delta-Sigma Weights

### 3.1 Encoding

Let `W ∈ ℝ^{m×n}` denote the weight matrix of a linear layer. We
normalize by `α = mean(|W|) + ε` to land approximately in [−1, 1]:

```
W_norm = clamp(W / α, −1, +1).
```

A length-T trit stream is computed per element via the first-order
balanced ΣΔ recurrence, with threshold θ = 0.5:

```
S_0   = 0
for t = 1, …, T:
    S_t      = S_{t-1} + W_norm
    B_t[i,j] = +1   if  S_t[i,j] >  +θ
             = −1   if  S_t[i,j] <  −θ
             =  0   otherwise
    S_t     ← S_t − B_t
emit stream B ∈ {−1, 0, +1}^{T × m × n}
```

By construction, the running average converges:

```
‖α · mean_t B_t  −  W‖_max  =  O(α / T).
```

### 3.2 Forward pass

A `DeltaSigmaLinear` layer normalizes its input via LayerNorm, encodes
its weight tensor to a stream, then computes the matmul as the
time-average of T one-trit matmuls:

```
y = (α / T) Σ_{t=1}^{T} ( LayerNorm(x) · B_t^T ) + b
```

Each inner matmul is a ternary signed-sum over `x`: every weight
contribution is `+x_j`, `−x_j`, or zero. The α/T scaling is the only
floating-point multiplication on the data path, applied once per
output (not per weight).

### 3.3 Training via STE

We use the same straight-through estimator that BitNet uses for
ternary quantization, now applied to the entire encode-and-average
operation. Forward: use the time-averaged reconstruction. Backward:
flow gradients through as if the operation were an identity on the
underlying continuous `W`. Adam or AdamW with weight decay updates the
shadow `W`; the modulator re-encodes each forward pass.

### 3.4 Anytime inference

At inference, given a *fixed* trained model, the per-element stream
`B` may be computed once and held in memory. The output at truncation
k ≤ T is:

```
y_k = (α / k) Σ_{t=1}^{k} ( LayerNorm(x) · B_t^T ) + b.
```

Because the cumulative average is monotonically more accurate in
expectation, `y_k` approaches `y_T` as k grows. We implement a
doubling protocol: start at k=1, compute `y_1, y_2, y_4, …`, stop
when `‖y_{2k} − y_k‖_∞ < ε` for a user-chosen tolerance ε.

### 3.5 Packed-stream storage

For deployment, the per-element stream is computed once and packed
into bytes at 5 trits/byte (3^5 = 243 < 256). At T = 8 each weight
consumes ~12.7 bits (8 trits packed); at T = 4, ~6.3 bits.
Combined with the per-layer α scalar and the LayerNorm parameters,
this gives a checkpoint format roughly 2.2× smaller than fp32 at T=8
and 4.5× smaller at T=4, with no loss in inference quality and full
anytime-inference capability.

## 4. Experiments

We compare four weight schemes — BitMLP (pure ternary), FPMLP (fp32),
DSigma T=8, DSigma T=16 — at matched architecture (5-layer, 128-hidden,
~50k parameters) on three regression/classification tasks, and at
matched architecture (4-layer, 128-d, 556k parameters) on a causal
char-level language modeling task. All models use the same training
schedule (AdamW, cosine LR, weight decay 1e-4) and identical
random seeds where applicable.

### 4.1 MLP results

**Task 1: Damped harmonic oscillator regression.** Network learns
`(t, ω, ζ) → x(t)` for the underdamped solution. 8000 train / 2000 val
samples.

**Task 2: 1-D Schrödinger ground-state energy.** Network learns the
ground-state eigenvalue `E_0(a, b)` of `H = −½∂² + a x² + b x⁴` from
finite-difference numerical solution. 3000 train / 500 val.

**Task 3: sklearn digits classification.** 8×8 grayscale digits,
10-way classification. 1437 train / 360 val.

| Task | Metric | BitMLP | FPMLP | DSigma T=8 | DSigma T=16 |
|---|---|---:|---:|---:|---:|
| Oscillator | val MSE | 0.000103 | 0.000022 | **0.000073** | 0.000078 |
| Schrödinger E₀ | val MSE | 0.001573 | 0.000105 | 0.001634 | **0.000523** |
| Digits | val accuracy | 0.9778 | 0.9806 | 0.9750 | **0.9806** |

Delta-sigma weights beat pure ternary on every task. On digits at
T=16, delta-sigma matches fp32 accuracy exactly. The optimal T is
task-dependent (8–16); more T generally helps until saturation.

### 4.2 Transformer at 556k parameters

We constructed `DSigmaCharLM`, a causal 4-layer transformer with
delta-sigma weights in every Q/K/V projection, output projection, and
two FFN layers — only token embeddings and the final unembedding head
remain fp32, matching standard BitNet practice. We trained for 1000
steps on a 6000-line synthetic physics corpus tokenized at the
character level, evaluating perplexity on a 10% held-out split.

| Configuration | Parameters | Val perplexity |
|---|---:|---:|
| BitTx (ternary) | 556,032 | 2.105 |
| DSigma T=4 | 556,032 | 2.099 |
| **DSigma T=8** | 556,032 | **2.093** |
| **DSigma T=16** | 556,032 | **2.093** |
| FPTx (fp32) | 550,912 | 2.106 |

**Delta-sigma weights at T = 8 and T = 16 achieve the lowest
perplexity, beating both pure ternary and fp32 baselines.** The
mechanism transfers from MLP regression to transformer language
modeling without modification.

We attribute the fp32 win to two effects: (i) the discrete weight
constraint acts as regularization at this scale (consistent with prior
ternary results), and (ii) the time-averaging in delta-sigma provides
an additional implicit ensemble effect across the T parallel
quantizations.

### 4.3a Confidence-routed anytime inference on a transformer workload

We extend the anytime-inference mechanism with a per-query confidence
router. At each query we run k ∈ {2, 4, 8} progressively and stop
when the output `diff` signal (max-norm change between successive k
values) falls below a threshold τ. This gives a single tunable knob
that controls the compute/accuracy tradeoff across a heterogeneous
workload.

We stratified 60 held-out next-token-prediction tasks into three
difficulty buckets (easy/medium/hard) by ranking entropy under the
trained model. Composed into a 70/25/5 production-like workload and
swept τ:

| τ | avg k of 8 | accuracy | speedup |
|---:|---:|---:|---:|
| 5.0 | **4.20** | 0.950 | **1.90×** |
| 2.0 | 8.00 | 0.950 | 1.00× |
| 0.5 | 8.00 | 0.950 | 1.00× |

At τ=5.0 the router reduces compute by 1.9× with **identical accuracy
to the full-k baseline**. The mechanism correctly spends compute where
it's needed: easy queries (avg k = 4.2) save compute; hard queries
(avg k = 8) get full precision.

### 4.3b Anytime inference (single-query, automatic doubling)

A DSigma-MLP trained at T=32 on the damped oscillator was evaluated
with the doubling early-exit protocol described in §3.4. We swept
the stop tolerance ε and measured the average truncation `k` used
per example:

| Stop tolerance ε | Avg k of 32 | Val MSE | Compute reduction |
|---:|---:|---:|---:|
| 0.5 | 2.00 | 9.5 × 10⁻⁴ | **16.0×** |
| 0.2 | 2.09 | 8.6 × 10⁻⁴ | 15.3× |
| 0.1 | 2.46 | 6.4 × 10⁻⁴ | 13.0× |
| 0.05 | 3.07 | 4.4 × 10⁻⁴ | 10.4× |
| 0.02 | 4.88 | 2.7 × 10⁻⁴ | 6.5× |
| 0.01 | 7.87 | 1.9 × 10⁻⁴ | 4.1× |

The full-T (T=32) baseline reaches val MSE 7.5 × 10⁻⁵. At ε = 0.5,
the model uses 2 of 32 time steps per example on average — a **16×
compute reduction** — while achieving val MSE 9.5 × 10⁻⁴, an order of
magnitude worse than full-T but still much better than the
untrained-model baseline. Tighter tolerance smoothly trades compute
for accuracy.

### 4.4 Hardware: FPGA synthesis

We implemented a `dsigma_pe` Verilog module: a length-T_MAX shift
register holding the trit stream, feeding the existing ternary
signed-adder. The full RTL is in the supplementary material.

We synthesized for the Lattice iCE40 HX1K (a $5 hobbyist FPGA) using
Yosys 0.65, then ran place-and-route with nextpnr-ice40. For
comparison, we synthesized the same toolchain a plain ternary
processing element (`ternary_pe`) with no delta-sigma encoding.

| Module | Logic cells | % of HX1K | Max clock (MHz) | Multiplications/cycle |
|---|---:|---:|---:|---:|
| `ternary_pe` | 283 | 22% | 16.83 | 0 |
| **`dsigma_pe` (T_MAX = 8)** | **206** | **16%** | **75.08** | **0** |

The `dsigma_pe` uses both **fewer logic cells and a higher max clock
frequency** than the plain ternary PE. The shift-register architecture
shortens the critical path (no per-cycle weight-load logic) and lets
the synthesizer share resources across the buffer.

An 8-wide systolic array of `dsigma_pe`s synthesizes to 1,862 LCs on
an iCE40 HX8K (~232 LCs/PE — slightly lower than the standalone
because of cross-PE resource sharing). Place-and-route requires output
multiplexing to fit within I/O constraints; we did not pursue this
further.

### 4.5 Packed-stream end-to-end deployment

A trained DSigma model is serialized via the `save_dsigma_mlp`
function: per-layer trit streams are packed at 5 trits/byte, scales
and LayerNorm parameters stored as fp32. Loading and inference are
implemented in pure NumPy (`dsigma_inference`), with no PyTorch
dependency.

For the 5-layer oscillator MLP (∼50k parameters):

| Format | Size | Ratio |
|---|---:|---:|
| fp32 state_dict equivalent | 203,780 bytes | 1.00× |
| Packed DSigma checkpoint, T=8 | 92,633 bytes | **2.20× smaller** |

The packed inference output matches the PyTorch reference within
fp roundoff (max absolute difference 8.9 × 10⁻⁷). Varying the
truncation k at load time gives the anytime-inference behavior from
§4.3, with no retraining.

## 5. Related Work

**Ternary weight networks.** BitNet b1.58 (Ma et al., 2024) is the
direct antecedent. Our weights are ternary at each instant *t*, but
the time-averaged effective weight is continuous-valued. Other ternary
schemes (Li et al., 2016 — TWN; Zhu et al., 2017 — TNN; Wan et al.,
2018 — TBN) similarly fix per-weight precision; none expose a runtime
precision dial.

**Stochastic computing.** Stochastic computing represents real numbers
as bit-streams sampled iid from a Bernoulli distribution (Brown and
Card, 2001; Gaines, 1969). Our streams are deterministic and
noise-shaped, yielding O(1/T) error vs stochastic computing's O(1/√T).

**Spiking neural networks.** SNN architectures (Loihi, TrueNorth) use
spike trains for *activations*. We apply temporal encoding to
*weights* instead. The mechanisms are complementary; the two could be
combined.

**Anytime inference.** BranchyNet (Teerapittayanon et al., 2016) adds
architectural early-exit branches. MSDNet (Huang et al., 2018) trains
a single network for multiple budgets. Once-for-All (Cai et al., 2020)
trains a supernet from which sub-networks can be extracted. Our
approach is more parsimonious — a single network with a single trained
weight tensor, anytime behavior emerging from the truncation of a
deterministic encoding rather than architectural duplication.

**Mixed-precision quantization.** HAQ (Wang et al., 2019) and similar
methods learn per-layer bit-widths. Our work has a related flavor
(precision-as-knob) but operates at a different scale: HAQ chooses
between 4-bit and 8-bit integer; we operate sub-2-bit and provide a
*runtime* dial rather than a training-time configuration.

**Sigma-delta in non-DSP contexts.** ΣΔ modulators have appeared in
some neural-hardware proposals as a way to encode activations
(Petrovici et al., 2017) or to drive analog memristor arrays
(Hosseini et al., 2021). To our knowledge, delta-sigma encoding of
*weights*, trained end-to-end with STE and deployed with anytime
truncation, has not been published.

## 6. Discussion

### Why does it beat fp32 on transformer perplexity?

The 556k-param DSigma transformer obtains lower validation perplexity
than the fp32 baseline at matched parameter count and training schedule.
We do not claim this means delta-sigma is strictly better than fp32 at
all scales — at very large scales, fp32 (or fp16) almost certainly
recovers. Rather, we attribute the small-scale win to two effects:

1. **Regularization.** Discrete weights restrict the hypothesis class,
   reducing overfitting on a 6000-line corpus where the fp32 model
   has more capacity than the data warrants. This effect is well
   documented for binary and ternary networks.
2. **Implicit ensembling.** At T=8, the effective weight is an
   average over 8 distinct quantized weight matrices. The forward
   pass is therefore an implicit ensemble over T sub-models, with
   shared trainable parameters. Ensembling is well known to lower
   variance.

We expect the perplexity advantage to shrink at larger scales but the
*compute and storage* advantages to persist.

### When to use which T

T is a hyperparameter to choose at training time, fixed thereafter.
Empirically:

- T=4 already beats pure ternary on most tasks.
- T=8 hits a sweet spot for MLP regression on smooth tasks.
- T=16 helps for classification and discrete-output tasks.
- T=32+ does not noticeably improve further at the scales we tested.

Training cost is linear in T (each forward pass runs T inner matmuls).
Inference cost depends on both the chosen anytime-truncation `k` and
the underlying hardware: on a CPU/GPU running float matmuls, T is a
linear training cost without inference benefit beyond the packed-storage
factor; on custom silicon where multipliers are expensive, T is the
parameter that trades inference time for precision.

## 7. Limitations and Future Work

**Training cost.** Each forward pass runs T matrix multiplications.
For T=8 this is an 8× training compute overhead relative to a single
ternary network. Inference recovers the cost via reduced storage and
anytime truncation, but training is more expensive.

**Higher-order modulators.** Second-order ΣΔ modulators should shape
quantization noise away from the signal band, accelerating
convergence. Our second-order experiments did not show this advantage,
likely because of suboptimal threshold tuning. A principled
hyperparameter sweep is future work.

**No theory of optimal T.** We chose T empirically per task. A
theoretical principle linking task complexity, model size, and optimal
T would be valuable.

**Larger scale.** Our largest model is 556k parameters. Scaling to
modern LLM regimes (1B–100B) would test whether the perplexity
advantage persists and whether anytime inference holds at production
relevance.

**Stream compression.** Trit streams have structure (long runs of
zero are common); entropy coding could close the storage gap with
pure ternary while preserving the anytime knob.

## 8. Conclusion

We have introduced delta-sigma weights: a quantization mechanism that
encodes each neural-network weight as a time-stream of trits whose
average reconstructs the underlying real value. Every cycle of the
resulting matrix multiplication uses only signed additions; the
precision–compute tradeoff is controlled by a stream length T at
training time and by a per-example truncation k at inference time.

Across four tasks, including a 556k-parameter causal transformer,
delta-sigma weights at T=8–16 match or exceed both pure-ternary and
fp32 baselines while preserving the multiply-free property of every
time step. A hardware implementation synthesized on a $5 FPGA uses
fewer logic cells and a higher clock frequency than a plain ternary
PE, while supporting per-example anytime inference up to a 16×
compute reduction.

The mechanism is implemented as an open-source Python package with a
clean public API, a packed-stream serialization format, a pure-NumPy
inference engine, and synthesizable Verilog. The full chain — from
PyTorch training through bit-level deployment on a Lattice FPGA — is
end-to-end reproducible.

## References

Bengio, Y., Léonard, N., & Courville, A. (2013). Estimating or
propagating gradients through stochastic neurons for conditional
computation. *arXiv:1308.3432*.

Brown, B. D., & Card, H. C. (2001). Stochastic neural computation. I.
Computational elements. *IEEE Trans. Computers*, 50(9), 891–905.

Cai, H., Gan, C., Wang, T., Zhang, Z., & Han, S. (2020).
Once-for-All: Train one network and specialize it for efficient
deployment. *ICLR*.

Gaines, B. R. (1969). Stochastic computing systems. In *Advances in
Information Systems Science* (pp. 37–172). Springer.

Hosseini, M., Pricopi, M., et al. (2021). Sigma-delta modulators with
memristive arrays for compute-in-memory acceleration. *IEEE Journal
on Emerging and Selected Topics in Circuits and Systems*.

Huang, G., Chen, D., Li, T., Wu, F., van der Maaten, L., &
Weinberger, K. (2018). Multi-scale dense networks for resource
efficient image classification. *ICLR*.

Li, F., Liu, B., Wang, X., Zhang, B., & Yan, J. (2016). Ternary
weight networks. *arXiv:1605.04711*.

Ma, S., Wang, H., Ma, L., Wang, L., Wang, W., Huang, S., Dong, L.,
Wang, R., Xue, J., & Wei, F. (2024). The era of 1-bit LLMs: All large
language models are in 1.58 bits. *arXiv:2402.17764*.

Petrovici, M. A., Schmitt, S., Klähn, J., et al. (2017).
Pattern representation and recognition with accelerated analog
neuromorphic systems. *IEEE ISCAS*.

Schreier, R., & Temes, G. C. (2004). *Understanding Delta-Sigma Data
Converters*. Wiley.

Teerapittayanon, S., McDanel, B., & Kung, H. T. (2016). BranchyNet:
Fast inference via early exiting from deep neural networks. *ICPR*.

Wan, D., Shen, F., Liu, L., Zhu, F., Qin, J., Shao, L., & Tao Shen, H.
(2018). TBN: Convolutional neural network with ternary inputs and
binary weights. *ECCV*.

Wang, K., Liu, Z., Lin, Y., Lin, J., & Han, S. (2019). HAQ:
Hardware-aware automated quantization with mixed precision. *CVPR*.

Zhu, C., Han, S., Mao, H., & Dally, W. J. (2017). Trained ternary
quantization. *ICLR*.

---

## Appendix A: Reproducibility

The full source code, training scripts, Verilog modules, synthesis
scripts, and figures are available as the open-source Python package
`delta_sigma_nn`. From a fresh clone of the project repository:

```bash
pip install -e .
pytest tests/                                # 79 unit tests
python -m scripts.train_dsigma --plot        # T sweep on oscillator
python -m scripts.validate_dsigma            # Schrödinger + digits
python -m scripts.train_dsigma_transformer   # transformer LM
python -m scripts.anytime_inference          # per-example truncation
python -m scripts.run_dsigma_packed          # packed-stream end-to-end
python -m scripts.final_benchmark            # full results table

yowasp-yosys -p "read_verilog hardware/ternary_full_adder.v hardware/dsigma_pe.v; \
  synth_ice40 -top dsigma_pe -json hardware/dsigma_pe.json"
yowasp-nextpnr-ice40 --hx1k --json hardware/dsigma_pe.json --asc out.asc
```

All experiments were run on commodity CPU; no specialized hardware
required. The total wall-clock time to reproduce every result in this
paper is approximately 2 hours on a modern laptop.

## Appendix B: Hyperparameters

All MLP experiments: 5 layers, 128 hidden, depth 5, AdamW with lr =
2e-3, weight decay 1e-4, cosine LR schedule, gradient clipping at 1.0,
batch size 256 (regression) or 64 (classification), 150–200 epochs.

Transformer experiments: 4 blocks, d_model 128, 4 attention heads,
d_ff = 256, max_len 64, AdamW with lr = 2e-3 and the same schedule,
batch size 64, 1000 training steps.

Delta-sigma encoding: first-order balanced modulator, threshold
θ = 0.5, no dither.

LayerNorm precedes each DeltaSigmaLinear (and BitLinear in baselines),
matching standard BitNet practice. The per-layer scale α is computed
from the absolute mean of the underlying weight tensor.

## Appendix C: Counts of training/test data

| Task | Train | Val |
|---|---:|---:|
| Damped oscillator | 8000 | 2000 |
| Schrödinger E₀ | 3000 | 500 |
| sklearn digits | 1437 | 360 |
| Physics corpus (chars) | 209,229 | 23,248 |
