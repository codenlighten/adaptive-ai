# Contributing to adaptive-ai

Thanks for your interest. This repo started as a research exploration and is
becoming a useful library — both halves welcome contributions.

## Quick start

```bash
git clone https://github.com/codenlighten/adaptive-ai
cd adaptive-ai
python3 -m venv venv && source venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
python -m pytest tests/
```

If all 83+ tests pass, you're set up.

## Repository structure

| path | purpose |
|---|---|
| `delta_sigma_nn/` | pip-installable library — the productized core mechanism |
| `src/` | research codebase (HNN, EQL, MoE, BPE, diffusion, Setun VM, etc.) |
| `scripts/` | reproducible demos and experiments |
| `tests/` | unit tests (run with `pytest tests/`) |
| `hardware/` | Verilog modules, synthesis scripts, FPGA results |
| `PAPER.md` / `DELTA_SIGMA_WEIGHTS.md` | technical writeups |

When making changes:
- Library code goes in `delta_sigma_nn/`. Add tests in `tests/`.
- Research/experiment code goes in `src/` and `scripts/`. Keep these
  independent from the library — they can import the library, but the
  library should never depend on them.

## Areas where we'd love contributions

### Production-facing
- **LLM-scale validation.** Train DSigma transformers at 1B+ params on a
  real corpus and report perplexity vs T, vs BitNet, vs fp16.
- **Vectorized encoders.** The Python `encode_delta_sigma_*` loops are
  the bottleneck. A Triton or CUDA kernel would unlock real-time
  confidence routing.
- **Real workload benchmarks.** Replace synthetic physics text with
  ShareGPT, OpenAssistant, MMLU, HellaSwag.
- **Inference routing infrastructure.** A real request scheduler that
  exposes the runtime precision knob with SLA controls and quality
  monitoring.

### Library polish
- **Type hints** throughout the public API.
- **Tutorial notebooks** demonstrating each piece.
- **Benchmark suite** against fp16/INT8/INT4/BitNet baselines on
  standard hardware.

### Hardware
- **Larger systolic arrays.** Our 8-PE array synthesizes; place-and-route
  hits I/O limits. A version with shared output buses would scale.
- **Real FPGA bring-up.** We've gone as far as nextpnr; the next step
  is generating actual iCE40 bitstream and demoing on a $5 board
  (TinyFPGA, iCEBreaker).

### Research extensions
- **Stochastic encoding** variant for unbiased low-T inference.
- **Adaptive T** per layer learned during training.
- **Combination with KV-cache quantization** for transformers.
- **Stream entropy coding** to close the storage gap with pure ternary.

## Pull request guidelines

1. **One thing per PR.** Easier to review and revert.
2. **Tests for new code.** If you add a public function, add a unit test
   that exercises it.
3. **No formatter required.** Standard PEP 8 is fine. Don't reformat
   unrelated code.
4. **Use existing import patterns.** Library modules: relative imports
   inside `delta_sigma_nn/`. Tests/scripts: `from delta_sigma_nn.X
   import Y`. Research code in `src/`: relative imports within `src/`.

## Issue guidelines

Bug reports — tell us:
- What you tried (a minimal reproduction is gold)
- What you expected
- What happened instead
- Versions: `python --version`, `pip show delta-sigma-nn | head`

Feature requests — tell us:
- The use case (what are you trying to do?)
- Why it's not possible with the current API
- How you imagine the new behavior

Discussion topics are also welcome — open an issue with the `discussion`
label.

## Code of conduct

Be respectful. Critique work, not people. We are a small team and want
to keep collaboration friction-free.

## License

By contributing to this repository, you agree your contributions will be
licensed under the MIT License (see `LICENSE`).
