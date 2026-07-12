"""
evaluate_ref.py
---------------
Reference-band detection: compares each failure condition against its
clean-training control (same data pipeline, same architecture) instead of
against its own first epoch.

Rationale: the first-change metric in evaluate.py fires within the first two
epochs on perfectly healthy training runs (ordinary warm-up dynamics), and its
min-max normalization uses the whole run (future information). Detection here
is causal and deployable: for a monitored signal x(t) and a clean-reference
band (mu_t, sd_t) built from N clean runs, the detection epoch is the first t
with |x(t) - mu_t| > k * sd_t.

Reported per (condition, signal):
  - detection epoch at k = 2 and k = 3 (mean +/- std over failure runs)
  - false-alarm rate: leave-one-out detection on the clean runs themselves
  - effect size: |z| at epoch 1 and max |z| over training (mean over runs)

Usage:
    python evaluate_ref.py --results results/
"""

import argparse

import numpy as np

from evaluate import load_runs, METRICS, METRIC_LABELS, CONDITION_LABELS

# failure condition -> its clean-training reference (identical eval pipeline)
PAIRS = {
    "label_noise":     "clean_mnist",
    "spurious":        "clean_cifar_raw",
    "class_imbalance": "clean_cifar",
    "dist_shift":      "clean_resnet",
}


def band(clean_runs, key):
    """Per-epoch mean and std of a metric across clean runs -> (mu, sd)."""
    C = np.array([m[key] for m in clean_runs], dtype=np.float64)
    return C.mean(axis=0), C.std(axis=0) + 1e-12


def detect(x, mu, sd, k):
    """First epoch (1-indexed) where x leaves the k-sigma band, else None."""
    x = np.asarray(x, dtype=np.float64)
    t = min(len(x), len(mu))
    out = np.where(np.abs(x[:t] - mu[:t]) > k * sd[:t])[0]
    return int(out[0]) + 1 if len(out) else None


def false_alarm_rate(clean_runs, key, k):
    """Leave-one-out: how often does a held-out CLEAN run trip the band?"""
    if len(clean_runs) < 3:
        return float("nan")
    fired = 0
    for i in range(len(clean_runs)):
        rest = [m for j, m in enumerate(clean_runs) if j != i]
        mu, sd = band(rest, key)
        if detect(clean_runs[i][key], mu, sd, k) is not None:
            fired += 1
    return fired / len(clean_runs)


def fmt_detect(epochs, horizon):
    fired = [e for e in epochs if e is not None]
    n = len(epochs)
    if not fired:
        return f"never (>{horizon})"
    s = f"{np.mean(fired):.1f} +/- {np.std(fired):.2f}"
    if len(fired) < n:
        s += f" ({len(fired)}/{n} runs)"
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results")
    args = parser.parse_args()

    print("\n" + "=" * 86)
    print("REFERENCE-BAND DETECTION - first epoch outside the clean-training band")
    print("(causal: uses only clean-reference statistics available at epoch t)")
    print("=" * 86)

    for cond, ref in PAIRS.items():
        fail_runs = load_runs(args.results, cond)
        clean_runs = load_runs(args.results, ref)
        if not fail_runs:
            print(f"\n{CONDITION_LABELS[cond]}: no failure runs found, skipping.")
            continue
        if len(clean_runs) < 3:
            print(f"\n{CONDITION_LABELS[cond]}: need >= 3 reference runs of '{ref}', "
                  f"found {len(clean_runs)} (python train.py --condition {ref}), skipping.")
            continue

        horizon = min(len(m["accuracy"]) for m in fail_runs + clean_runs)
        print(f"\n{CONDITION_LABELS[cond]}   [reference: {ref}, "
              f"{len(clean_runs)} clean runs, horizon {horizon} epochs]")
        header = (f"  {'Signal':<15} {'detect@2s':>20} {'FA@2s':>7} "
                  f"{'detect@3s':>20} {'FA@3s':>7} {'|z| ep1':>9} {'max |z|':>9}")
        print(header)
        print("  " + "-" * (len(header) - 2))

        for key in METRICS:
            mu, sd = band(clean_runs, key)
            det2 = [detect(m[key], mu, sd, 2) for m in fail_runs]
            det3 = [detect(m[key], mu, sd, 3) for m in fail_runs]
            fa2 = false_alarm_rate(clean_runs, key, 2)
            fa3 = false_alarm_rate(clean_runs, key, 3)
            z = np.array([np.abs((np.asarray(m[key][:horizon]) - mu[:horizon]) / sd[:horizon])
                          for m in fail_runs])
            print(f"  {METRIC_LABELS[key]:<15} {fmt_detect(det2, horizon):>20} {fa2:>7.2f} "
                  f"{fmt_detect(det3, horizon):>20} {fa3:>7.2f} "
                  f"{z[:, 0].mean():>9.1f} {z.max(axis=1).mean():>9.1f}")

    print("\nNotes: FA = leave-one-out false-alarm rate on clean runs (lower is better;")
    print("a detector is only meaningful alongside it). |z| ep1 = deviation from the")
    print("clean band at the first epoch - the effect size an operator would see.\n")


if __name__ == "__main__":
    main()
