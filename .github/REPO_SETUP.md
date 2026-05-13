# GitHub repo configuration (manual UI steps)

These are settings that need to be made in the GitHub web UI at
https://github.com/codenlighten/adaptive-ai/settings — they can't be set
from the command line.

## Description (Settings → General → Description)

> Multiply-free neural networks with delta-sigma weights and anytime inference. Runtime precision knob for adaptive AI compute.

## Topics (front page → ⚙ next to "About")

```
quantization
neural-networks
bitnet
delta-sigma
anytime-inference
adaptive-inference
ternary-neural-networks
fpga
verilog
efficient-inference
edge-ai
pytorch
```

## Settings recommendations

- **Default branch**: `main` ✓ (already set)
- **Issues**: enabled ✓
- **Discussions**: enable for design conversations
- **Wikis**: probably disable (use `docs/` or README instead)
- **Sponsorships**: optional
- **Preserve this repository**: optional (Software Heritage / Arctic Vault)

## Branch protection (Settings → Branches → Add rule for `main`)

Recommended:
- Require a pull request before merging
- Require status checks to pass before merging
  - Require branches to be up to date before merging
  - Check status: `tests` (will appear once first CI run completes)
- Require linear history
- Allow force pushes: **disabled**
- Allow deletions: **disabled**

## Social preview image (Settings → General → Social preview)

Recommended: a clean ~1280×640 image showing one of the result plots
(e.g., `dsigma_results.png` or `rate_distortion.png`). GitHub uses
this for link previews on Twitter/Slack/etc.

## Pages (Settings → Pages)

Once the paper is rendered as HTML, you can host it at
`https://codenlighten.github.io/adaptive-ai/` by enabling Pages from
the `main` branch's `/docs` folder.
