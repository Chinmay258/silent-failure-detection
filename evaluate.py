"""
evaluate.py
-----------
Loads saved metrics from all runs and conditions, computes lead-time
statistics, and prints the result table for the paper.

Usage:
    python evaluate.py --results results/
"""

import argparse
import os
import numpy as np


CONDITIONS = ["label_noise", "spurious", "dist_shift", "class_imbalance"]
CONDITION_LABELS = {
    "label_noise":     "Label Noise (MNIST)",
    "spurious":        "Spurious Features (CIFAR-10)",
    "dist_shift":      "Distribution Shift (CIFAR-10 / ResNet-18)",
    "class_imbalance": "Class Imbalance (CIFAR-10)",
}
METRICS = ["accuracy", "confidence", "dispersion", "effective_rank"]
METRIC_LABELS = {
    "accuracy":       "Accuracy",
    "confidence":     "Confidence",
    "dispersion":     "Dispersion",
    "effective_rank": "Eff. Rank",
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
    return len(x)   # never changed → assign max (worst) epoch


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


def compute_lead_times(all_metrics):
    """Returns dict: metric -> list of first_change epochs across runs."""
    lead = {k: [] for k in METRICS}
    for m in all_metrics:
        for k in METRICS:
            lead[k].append(first_change(m[k]))
    return lead


def summarize(lead):
    """Returns dict: metric -> (mean, std)."""
    return {
        k: (np.mean(v), np.std(v))
        for k, v in lead.items()
    }


def print_table(results_dir):
    print("\n" + "=" * 70)
    print("LEAD-TIME ANALYSIS — First epoch of significant signal change")
    print(f"(threshold = {THRESH} normalized units, lower = reacts sooner)")
    print("=" * 70)

    header = f"{'Condition':<36} {'Metric':<14} {'Mean':>6}  {'±Std':>5}"
    print(header)
    print("-" * 70)

    geometry_wins = []   # store (condition, geometry_metric, lead_epochs)

    for cond in CONDITIONS:
        all_metrics = load_runs(results_dir, cond)
        if not all_metrics:
            print(f"  {CONDITION_LABELS[cond]}: no data found, skipping.")
            continue

        lead    = compute_lead_times(all_metrics)
        summary = summarize(lead)

        first = True
        for k in METRICS:
            mean, std = summary[k]
            cond_label = CONDITION_LABELS[cond] if first else ""
            print(f"{cond_label:<36} {METRIC_LABELS[k]:<14} {mean:>6.1f}  ±{std:>4.2f}")
            first = False

        # compute geometry advantage
        acc_mean = summary["accuracy"][0]
        for gk in ["dispersion", "effective_rank"]:
            lead_epochs = acc_mean - summary[gk][0]
            geometry_wins.append((CONDITION_LABELS[cond], METRIC_LABELS[gk], lead_epochs))

        print()

    print("=" * 70)
    print("\nGEOMETRY LEAD OVER ACCURACY (positive = geometry fires first)")
    print("-" * 70)
    for cond_label, metric, lead_e in geometry_wins:
        direction = "earlier" if lead_e > 0 else "later"
        print(f"  {cond_label}")
        print(f"    {metric}: {abs(lead_e):.1f} epochs {direction} than accuracy")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results")
    args = parser.parse_args()
    print_table(args.results)


if __name__ == "__main__":
    main()
