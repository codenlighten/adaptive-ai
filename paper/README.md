# Paper source (LaTeX)

The workshop-style LaTeX paper sources for "Delta-Sigma Weights".

## Building

You need a TeX distribution (TeX Live, MacTeX, MikTeX) with
`pdflatex` and `bibtex` available.

```bash
cd paper
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

Or, with `latexmk`:

```bash
cd paper
latexmk -pdf paper.tex
```

Output: `paper.pdf` (approximately 8 pages including references).

## Files

- `paper.tex` — main paper source
- `paper.bib` — BibTeX references
- `README.md` — this file

## Workshop venue conversion

The paper uses the standard `article` class. To convert for a specific
workshop venue (e.g., NeurIPS, ICLR), replace the preamble with the
venue's style file:

```latex
\documentclass{neurips_2025}
% or
\documentclass[iclr2025_conference]{article}
```

The body of the paper should be venue-agnostic. The author block uses
`authblk` which most venues support; for venues that require a
specific format, swap out the `\title{...}` and `\author{...}` lines.

## Page count

At 11pt, 1-inch margins, the paper renders to approximately 8 pages.
Most workshop venues allow 4-6 pages of main text + unlimited
references; trimming for those venues would require cutting one of:

- the related-work section (1 page)
- the discussion section (1 page)
- a subset of result tables

Best candidate sections to trim are §6 (Discussion) and §7
(Limitations) — together they are about 1.5 pages.
