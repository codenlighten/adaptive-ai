# Roadmap

This file tracks the high-leverage work items that haven't been
completed yet. Each item has a sketch of what success looks like, the
rough effort, and the dependency.

## Tier 1: validation that the mechanism scales

### LLM-scale validation (the most important open question)

**Goal**: train a DSigma transformer at $\ge$1B parameters on a real
corpus (subset of FineWeb, OpenWebText, or similar) and measure
val perplexity vs $T$, compared against:

- An fp16 baseline of the same architecture
- A BitNet b1.58 model of the same architecture
- An INT8 quantized fp16 model

Expected outcome: one of three results, all useful.

1. *Best case*: the small-scale fp32-beating result holds. DSigma at
   $T=8$ matches or beats both BitNet and INT8 on perplexity, with the
   anytime knob unlocked. This is the result that justifies production
   adoption.
2. *Likely case*: DSigma ties pure ternary on perplexity (slightly
   above fp16 / INT8), but the anytime inference provides $2$--$4\times$
   compute reduction on a heterogeneous workload at zero quality loss.
   Still strong.
3. *Worst case*: training is unstable or quality degrades sharply at
   scale. Then we learn the regime where DSigma stops working and the
   paper should be reframed around the small-scale and FPGA results
   only.

**Compute**: 8 H100s for ~3--7 days at 1B params. Roughly \$2k--5k of
cloud GPU.

**Engineering**: 2--3 weeks. Pre-existing pieces are sufficient
(\texttt{DSigmaCharLM} works); the new work is BPE tokenization at
scale, real corpus loading, distributed training setup, evaluation
on MMLU/HellaSwag.

**Dependency**: someone with GPU budget + 2--3 weeks of focused time.

### Vectorized encode kernel

**Goal**: replace the Python loop in
\texttt{delta\_sigma\_nn/delta\_sigma.py:encode\_delta\_sigma\_ternary}
with a Triton or CUDA kernel. Target: 100$\times$ speedup over current
implementation, making the confidence router practical for real-time
serving.

**Effort**: 1--2 weeks.

**Status**: identified, not started.

### Real-workload benchmarks

**Goal**: replace synthetic physics text with ShareGPT or
OpenAssistant. Measure the actual easy/medium/hard query distribution
in the wild, project achievable compute reduction from confidence
routing.

**Effort**: 1 week.

**Dependency**: ideally pairs with LLM-scale validation.

## Tier 2: hardware path forward

### Real iCE40 bitstream

**Goal**: take our nextpnr-routed design through `icepack` to a real
bitstream, flash a TinyFPGA-BX or iCEBreaker board, run a small
matmul on hardware, and post a video.

**Effort**: 1--2 weeks including board-procurement.

**Status**: nextpnr place-and-route succeeded; bitstream generation
remains.

### ASIC area estimate via OpenROAD

**Goal**: take the synthesizable Verilog through the open-source
ASIC flow (Yosys + OpenROAD) targeting a real foundry library
(SkyWater 130nm or GF180 PDK). Report area in $\mu\text{m}^2$ and
estimated frequency.

**Effort**: 2--3 weeks.

**Status**: requires PDK familiarity.

## Tier 3: library polish

### Type hints throughout the public API

All functions in `delta_sigma_nn/__init__.py` should have full type
annotations. Light effort, large impact on adoption.

### Tutorial notebooks

Three Jupyter notebooks:
1. Training a DeltaSigmaMLP from scratch.
2. Anytime inference and the confidence router.
3. Deploying with packed-trit streams.

### Integration with Hugging Face

A `DSigmaConfig` and `DSigmaModel` that fit the HF `transformers`
ecosystem so users can `AutoModelForCausalLM.from_pretrained(...)`
on a DSigma checkpoint.

### Benchmark suite

A standardized benchmark script that measures DSigma against
fp16/INT8/INT4/BitNet on a fixed set of tasks and reports
accuracy/compute/memory. Output: a CSV plus a leaderboard PNG.

## Tier 4: research extensions

### Stochastic encoding

A randomized $\Sigma\Delta$ variant for unbiased low-T inference;
trades determinism for cleaner statistical convergence.

### Per-layer adaptive $T$

Learn the optimal $T$ per layer during training rather than using a
single global $T$. Pair with mixed-precision research (HAQ etc.).

### Combined with KV-cache quantization

For transformer inference, the KV cache often consumes more memory
than the model weights. Combining DSigma weights with quantized KV
cache could substantially improve long-context inference.

### Entropy coding of streams

Trit streams have structure: long runs of zero are common. Apply
arithmetic coding or RLE to close the storage gap with pure ternary
($\sim$1.6 bits/weight) while preserving the anytime knob.

## How to pick what to work on

If you want **immediate impact**: vectorized encode kernel + real
workload benchmark. Both unlock the productionability claim.

If you want **the strongest result**: LLM-scale validation. This is
the gate that turns "interesting mechanism" into "deployable
technique".

If you want **a credible hardware story**: real iCE40 bitstream
demo. Photogenic and concrete.

If you want **adoption**: HF integration + type hints + notebooks.
