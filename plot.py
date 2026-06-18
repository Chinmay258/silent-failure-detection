"""
plot.py
-------
Produces one clean 1×3 multi-panel figure across all three failure conditions.
Mean ± std band shown across runs.

Usage:
    python plot.py --results results/ --out figures/
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONDITIONS = ["label_noise", "spurious", "dist_shift", "class_imbalance"]
CONDITION_LABELS = {
    "label_noise":     "Label Noise\n(MNIST + CNN)",
    "spurious":        "Spurious Features\n(CIFAR-10 + CNN)",
    "dist_shift":      "Distribution Shift\n(CIFAR-10 + ResNet-18)",
    "class_imbalance": "Class Imbalance\n(CIFAR-10 + CNN)",
}
METRICS = ["accuracy", "confidence", "dispersion", "effective_rank"]
METRIC_LABELS = {
    "accuracy":       "Accuracy",
    "confidence":     "Confidence",
    "dispersion":     "Dispersion",
    "effective_rank": "Effective Rank",
}
COLORS = {
    "accuracy":       "#2196F3",   # blue
    "confidence":     "#FF9800",   # orange
    "dispersion":     "#4CAF50",   # green
    "effective_rank": "#F44336",   # red
}
THRESH = 0.1


def normalize(x):
    x = np.array(x, dtype=np.float64)
    rng = x.max() - x.min()
    if rng < 1e-8:
        return np.zeros_like(x)
    return (x - x.min()) / rng


def first_change(x, thresh=THRESH):
    x = normalize(x)
    base = x[0]
    for i in range(1, len(x)):
        if abs(x[i] - base) > thresh:
            return i
    return None


def load_runs(results_dir, condition):
    run = 0
    all_metrics = []
    while True:
        path = os.path.join(results_dir, condition, f"run{run}", "metrics.npy")
        if not os.path.exists(path):
            break
        m = np.load(path, allow_pickle=True).item()
        all_metrics.append(m)
        run += 1
    return all_metrics


def plot_condition(ax, all_metrics, condition):
    if not all_metrics:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        return

    epochs = len(all_metrics[0]["accuracy"])
    x = np.arange(1, epochs + 1)

    for metric in METRICS:
        # stack normalized runs
        stacked = np.stack([
            normalize(m[metric]) for m in all_metrics
        ], axis=0)                                   # (runs, epochs)
        mean = stacked.mean(axis=0)
        std  = stacked.std(axis=0)
        color = COLORS[metric]

        ax.plot(x, mean, label=METRIC_LABELS[metric],
                color=color, linewidth=2)
        ax.fill_between(x, mean - std, mean + std,
                        color=color, alpha=0.15)

        # mark first-change epoch (from mean signal)
        fc = first_change(mean)
        if fc is not None:
            fc_x = x[fc]
            ax.axvline(fc_x, color=color, linestyle="--",
                       linewidth=0.9, alpha=0.7)
            ax.text(fc_x + 0.05, 0.04, f"ep{fc_x}",
                    color=color, fontsize=6.5,
                    transform=ax.get_xaxis_transform(),
                    va="bottom", ha="left")

    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Normalized Value", fontsize=10)
    ax.set_title(CONDITION_LABELS[condition], fontsize=11, fontweight="bold")
    ax.set_ylim(-0.05, 1.2)
    ax.set_xlim(1, epochs)
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results")
    parser.add_argument("--out",     type=str, default="figures")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    fig, axes_grid = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    axes = axes_grid.flatten()
    fig.suptitle(
        "Silent Failure Detection via Prediction Geometry\n"
        "Shaded bands = ±1 std across runs  |  Dashed lines = first significant change",
        fontsize=12, y=1.01
    )

    for ax, cond in zip(axes, CONDITIONS):
        all_metrics = load_runs(args.results, cond)
        plot_condition(ax, all_metrics, cond)

    # single shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.04),
               fontsize=10, frameon=False)

    plt.tight_layout()
    out_path = os.path.join(args.out, "main_results.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved: {out_path}")


if __name__ == "__main__":
    main()
