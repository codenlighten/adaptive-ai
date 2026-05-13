"""Render benchmarks/results.csv into a leaderboard figure.

Reads the long-form CSV emitted by `scripts/benchmark_suite.py` and
produces a single PNG with one subplot per task. Each subplot shows
mean ± std across seeds for {fp32, INT8, BitNet, DSigma T=8, DSigma T=16}
at full inference precision. A second figure shows the anytime curve
for DSigma at increasing k.

Run:
    python -m scripts.render_leaderboard
    python -m scripts.render_leaderboard --csv benchmarks/results.csv \
        --out benchmarks/leaderboard.png
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_rows(csv_path: Path) -> list[dict]:
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def aggregate(rows: list[dict]) -> dict:
    """Group rows by (task, model_label, k) and compute mean / std over seeds."""
    bucket: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        task = r["task"]
        model = r["model"]
        T = r["T"]
        k = r["k"]
        if model == "dsigma":
            label = f"DSigma T={T}"  # k distinguishes the bucket; label stays the same
        elif model == "fp32":
            label = "fp32"
        elif model == "int8":
            label = "INT8"
        elif model == "bitnet":
            label = "BitNet"
        else:
            label = model
        key = (task, label, k or "full")
        bucket[key].append(float(r["value"]))

    out: dict[tuple, tuple[float, float]] = {}
    for key, vals in bucket.items():
        arr = np.array(vals)
        out[key] = (float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0)
    return out


def render(agg: dict, out_path: Path, anytime_out: Path) -> None:
    tasks_present = sorted({k[0] for k in agg.keys()})
    fig, axes = plt.subplots(1, len(tasks_present),
                             figsize=(4.0 * len(tasks_present), 4.2))
    if len(tasks_present) == 1:
        axes = [axes]
    full_order = ["fp32", "INT8", "BitNet", "DSigma T=8", "DSigma T=16"]
    colors = {"fp32": "#888", "INT8": "#5b9bd5", "BitNet": "#ed7d31",
              "DSigma T=8": "#70ad47", "DSigma T=16": "#264478"}
    for ax, task in zip(axes, tasks_present):
        labels = [m for m in full_order if (task, m, "full") in agg]
        means = [agg[(task, m, "full")][0] for m in labels]
        stds  = [agg[(task, m, "full")][1] for m in labels]
        xs = np.arange(len(labels))
        ax.bar(xs, means, yerr=stds, capsize=4,
               color=[colors[m] for m in labels], edgecolor="black", linewidth=0.5)
        is_acc = task == "digits"
        metric = "accuracy" if is_acc else "val MSE"
        ax.set_title(f"{task} ({metric})")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        if not is_acc:
            ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Delta-Sigma vs baselines (mean ± std over seeds)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")

    # Anytime curve: per DSigma config, plot value vs k.
    anytime_configs = sorted({(t, m) for (t, m, k) in agg.keys()
                              if m.startswith("DSigma") and k != "full"})
    if not anytime_configs:
        return

    tasks_with_anytime = sorted({t for (t, _) in anytime_configs})
    fig2, axes2 = plt.subplots(1, len(tasks_with_anytime),
                               figsize=(4.0 * len(tasks_with_anytime), 4.2))
    if len(tasks_with_anytime) == 1:
        axes2 = [axes2]
    for ax, task in zip(axes2, tasks_with_anytime):
        for label in ["DSigma T=8", "DSigma T=16"]:
            pts = [(int(k), agg[(task, label, k)][0], agg[(task, label, k)][1])
                   for (t, m, k) in agg.keys()
                   if t == task and m == label and k.isdigit()]
            pts.sort()
            if not pts:
                continue
            ks = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            es = [p[2] for p in pts]
            ax.errorbar(ks, ys, yerr=es, marker="o", capsize=3,
                        color=colors[label], label=label)
            # Horizontal line at the full-T result for that model.
            if (task, label, "full") in agg:
                ax.axhline(agg[(task, label, "full")][0], color=colors[label],
                           linestyle=":", alpha=0.5)
        is_acc = task == "digits"
        ax.set_title(f"{task} — anytime")
        ax.set_xlabel("k (truncation)")
        ax.set_ylabel("accuracy" if is_acc else "val MSE")
        if not is_acc:
            ax.set_yscale("log")
        ax.set_xscale("log", base=2)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig2.suptitle("Anytime inference: val metric vs truncation k", y=1.02)
    fig2.tight_layout()
    fig2.savefig(anytime_out, dpi=120, bbox_inches="tight")
    plt.close(fig2)
    print(f"wrote {anytime_out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="benchmarks/results.csv")
    parser.add_argument("--out", default="benchmarks/leaderboard.png")
    parser.add_argument("--anytime-out", default="benchmarks/anytime.png")
    args = parser.parse_args()

    rows = load_rows(Path(args.csv))
    agg = aggregate(rows)
    render(agg, Path(args.out), Path(args.anytime_out))


if __name__ == "__main__":
    main()
