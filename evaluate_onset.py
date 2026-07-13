"""
evaluate_onset.py
-----------------
Detection-delay analysis for mid-training failure onset.

Runs produced with `train.py --onset K` train clean for K epochs, then the
corruption switches on. Each signal is compared against the clean-reference
band (see evaluate_ref.py) at the calibrated threshold k* (worst clean
leave-one-out excursion, zero false alarms on the reference runs).

Reported per (condition, signal):
  - pre-onset false alarms: detections during the clean phase of the onset
    runs -- an out-of-sample validation of the calibrated threshold
  - detection delay: (first epoch outside the band after onset) - K,
    so a delay of 1 means "caught in the first corrupted epoch"

Usage:
    python evaluate_onset.py --results results/ --onset 10
"""

import argparse

import numpy as np

from evaluate import load_runs, METRICS, METRIC_LABELS, CONDITION_LABELS
from evaluate_ref import PAIRS, band, detect, calibrated_k


def first_outside_after(x, mu, sd, k, start):
    """First 1-indexed epoch > start outside the k-sigma band, else None."""
    x = np.asarray(x, dtype=np.float64)
    t = min(len(x), len(mu))
    out = np.where(np.abs(x[:t] - mu[:t]) > k * sd[:t])[0] + 1
    out = out[out > start]
    return int(out[0]) if len(out) else None


def fmt_delay(delays, horizon, onset):
    fired = [d for d in delays if d is not None]
    if not fired:
        return f"never (>{horizon - onset})"
    s = f"{np.mean(fired):.1f} +/- {np.std(fired):.2f}"
    if len(fired) < len(delays):
        s += f" ({len(fired)}/{len(delays)} runs)"
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results")
    parser.add_argument("--onset",   type=int, default=10,
                        help="clean epochs before corruption (must match "
                             "the --onset used for training)")
    args = parser.parse_args()
    K = args.onset

    print("\n" + "=" * 78)
    print(f"ONSET DETECTION - corruption switches on after epoch {K}")
    print("(threshold k* calibrated on the clean reference runs; delay of 1 =")
    print(" caught in the first corrupted epoch)")
    print("=" * 78)

    for cond, ref in PAIRS.items():
        onset_runs = load_runs(args.results, f"{cond}_onset{K}")
        clean_runs = load_runs(args.results, ref)
        if not onset_runs:
            print(f"\n{CONDITION_LABELS[cond]}: no runs found for "
                  f"'{cond}_onset{K}', skipping.")
            continue
        if len(clean_runs) < 3:
            print(f"\n{CONDITION_LABELS[cond]}: need >= 3 reference runs of "
                  f"'{ref}', found {len(clean_runs)}, skipping.")
            continue

        horizon = min(len(m["accuracy"]) for m in onset_runs + clean_runs)
        print(f"\n{CONDITION_LABELS[cond]}   [{len(onset_runs)} onset runs, "
              f"reference: {ref}, horizon {horizon} epochs]")
        header = (f"  {'Signal':<15} {'k*':>6} {'pre-onset FA':>13} "
                  f"{'detection delay':>24}")
        print(header)
        print("  " + "-" * (len(header) - 2))

        for key in METRICS:
            mu, sd = band(clean_runs, key)
            ks = calibrated_k(clean_runs, key, horizon)

            pre_fa = 0
            delays = []
            for m in onset_runs:
                first_any = detect(m[key], mu, sd, ks)
                if first_any is not None and first_any <= K:
                    pre_fa += 1
                d = first_outside_after(m[key], mu, sd, ks, K)
                delays.append(None if d is None else d - K)

            print(f"  {METRIC_LABELS[key]:<15} {ks:>6.1f} "
                  f"{pre_fa}/{len(onset_runs):>11} "
                  f"{fmt_delay(delays, horizon, K):>24}")

    print("\nNote: pre-onset FA counts detections during the clean phase of the")
    print("onset runs - fresh clean data the threshold was never calibrated on.\n")


if __name__ == "__main__":
    main()
