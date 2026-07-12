"""
plot_ref.py
-----------
For each failure condition, plots the four monitored signals (raw values, not
normalized) against the clean-training reference band from the matching
control condition. One 2x2 figure per condition.

Usage:
    python plot_ref.py --results results/ --out figures/
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluate import load_runs, METRICS, METRIC_LABELS, CONDITION_LABELS
from evaluate_ref import PAIRS

CLEAN_COLOR = "#2196F3"   # blue  (matches plot.py palette)
FAIL_COLOR  = "#F44336"   # red


def plot_pair(ax, clean_runs, fail_runs, key):
    horizon = min(len(m[key]) for m in clean_runs + fail_runs)
    x = np.arange(1, horizon + 1)
    for runs, color, label in ((clean_runs, CLEAN_COLOR, "Clean training"),
                               (fail_runs, FAIL_COLOR, "Failure condition")):
        A = np.array([m[key][:horizon] for m in runs], dtype=np.float64)
        mean, std = A.mean(axis=0), A.std(axis=0)
        ax.plot(x, mean, color=color, linewidth=2, label=label)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
    ax.set_title(METRIC_LABELS[key], fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_xlim(1, horizon)
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results")
    parser.add_argument("--out",     type=str, default="figures")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    for cond, ref in PAIRS.items():
        fail_runs = load_runs(args.results, cond)
        clean_runs = load_runs(args.results, ref)
        if not fail_runs or not clean_runs:
            print(f"{cond}: missing runs (failure: {len(fail_runs)}, "
                  f"clean '{ref}': {len(clean_runs)}), skipping.")
            continue

        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        fig.suptitle(
            f"{CONDITION_LABELS[cond]} vs. clean-training reference ({ref})\n"
            "Raw signal values - shaded bands = +/-1 std across runs",
            fontsize=12,
        )
        for ax, key in zip(axes.flatten(), METRICS):
            plot_pair(ax, clean_runs, fail_runs, key)

        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2,
                   bbox_to_anchor=(0.5, -0.02), fontsize=10, frameon=False)
        plt.tight_layout()
        out_path = os.path.join(args.out, f"ref_{cond}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Figure saved: {out_path}")


if __name__ == "__main__":
    main()
