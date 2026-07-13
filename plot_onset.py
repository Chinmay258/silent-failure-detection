"""
plot_onset.py
-------------
Figures for the mid-training onset experiment (train.py --onset K).

Per condition:
  onset<K>_<condition>.png           four monitored signals vs. the clean
                                     reference band, corruption start marked
  onset<K>_<condition>_baselines.png baseline / class-conditional signals
                                     from the onset runs (no reference bands
                                     yet -- descriptive)

Usage:
    python plot_onset.py --results results/ --out figures/ --onset 10
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
ONSET_COLOR = "#555555"

BASELINES = ["loss_gap", "grad_norm", "min_class_acc", "min_class_erank"]
BASELINE_LABELS = {
    "loss_gap":        "Test − train loss",
    "grad_norm":       "Gradient norm",
    "min_class_acc":   "Worst-class accuracy",
    "min_class_erank": "Worst-class eff. rank",
}


def series(m, key):
    if key == "loss_gap":
        return np.asarray(m["test_loss"]) - np.asarray(m["train_loss"])
    return np.asarray(m[key], dtype=np.float64)


def band_plot(ax, runs, key, color, label):
    A = np.array([series(m, key) for m in runs])
    x = np.arange(1, A.shape[1] + 1)
    mean, std = A.mean(axis=0), A.std(axis=0)
    ax.plot(x, mean, color=color, linewidth=2, label=label)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
    return A.shape[1]


def style(ax, horizon, onset, title):
    ax.axvline(onset + 0.5, color=ONSET_COLOR, linestyle="--", linewidth=1.2)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_xlim(1, horizon)
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results")
    parser.add_argument("--out",     type=str, default="figures")
    parser.add_argument("--onset",   type=int, default=10)
    args = parser.parse_args()
    K = args.onset

    os.makedirs(args.out, exist_ok=True)

    for cond, ref in PAIRS.items():
        onset_runs = load_runs(args.results, f"{cond}_onset{K}")
        clean_runs = load_runs(args.results, ref)
        if not onset_runs or not clean_runs:
            print(f"{cond}: missing runs (onset: {len(onset_runs)}, "
                  f"clean '{ref}': {len(clean_runs)}), skipping.")
            continue

        # -- monitored signals vs clean reference ---------------------------
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        fig.suptitle(
            f"{CONDITION_LABELS[cond]} - corruption switched on after epoch {K}\n"
            "Raw signal values vs. clean reference - dashed line = onset",
            fontsize=12,
        )
        for ax, key in zip(axes.flatten(), METRICS):
            band_plot(ax, clean_runs, key, CLEAN_COLOR, "Clean reference")
            horizon = band_plot(ax, onset_runs, key, FAIL_COLOR, f"Onset run")
            style(ax, horizon, K, METRIC_LABELS[key])
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2,
                   bbox_to_anchor=(0.5, -0.02), fontsize=10, frameon=False)
        plt.tight_layout()
        out_path = os.path.join(args.out, f"onset{K}_{cond}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Figure saved: {out_path}")

        # -- baselines (descriptive, onset runs only) ------------------------
        if all(k in onset_runs[0] for k in ("train_loss", "grad_norm")):
            fig, axes = plt.subplots(2, 2, figsize=(11, 8))
            fig.suptitle(
                f"{CONDITION_LABELS[cond]} - baseline signals around onset "
                f"(epoch {K})",
                fontsize=12,
            )
            for ax, key in zip(axes.flatten(), BASELINES):
                horizon = band_plot(ax, onset_runs, key, FAIL_COLOR, "Onset run")
                style(ax, horizon, K, BASELINE_LABELS[key])
            plt.tight_layout()
            out_path = os.path.join(args.out, f"onset{K}_{cond}_baselines.png")
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Figure saved: {out_path}")


if __name__ == "__main__":
    main()
