"""
Recompute every headline figure from the committed runs: calibrated thresholds,
detection epochs, margins, onset delays, loss baselines, the retracted v1
numbers, and repository storage. CPU only, runs in seconds.

Usage:
    python analysis/card_numbers.py
"""

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)

import numpy as np

from evaluate import load_runs, METRICS, first_change
from evaluate_ref import PAIRS, band, detect, calibrated_k, false_alarm_rate
from evaluate_onset import first_outside_after

print("== fixed-k leave-one-out false-alarm rates on the reference runs ==")
allfa = []
for cond, ref in PAIRS.items():
    clean = load_runs("results", ref)
    row = []
    for k in METRICS:
        f2, f3 = false_alarm_rate(clean, k, 2), false_alarm_rate(clean, k, 3)
        allfa += [f2, f3]
        row.append(f"{k}={f2:.1f}/{f3:.1f}")
    print(f"{cond:<16}", "  ".join(row))
print(f"range over conditions x signals x k in (2,3): {min(allfa):.1f}-{max(allfa):.1f}")

print("\n== calibrated k*, from-scratch detection epochs, margin ==")
for cond, ref in PAIRS.items():
    clean, fail = load_runs("results", ref), load_runs("results", cond)
    for k in METRICS:
        mu, sd = band(clean, k)
        ks = calibrated_k(clean, k, 20)
        det = [detect(m[k], mu, sd, ks) for m in fail]
        z = np.array([np.abs((np.asarray(m[k][:20]) - mu[:20]) / sd[:20]) for m in fail])
        print(f"{cond:<16} {k:<15} k*={ks:5.2f} det={det} "
              f"margin={(z.max(1) / ks).mean():.2f}")

print("\n== onset detection delays at k* (epochs after the switch) ==")
for cond, ref in PAIRS.items():
    clean, ons = load_runs("results", ref), load_runs("results", f"{cond}_onset10")
    for k in METRICS:
        mu, sd = band(clean, k)
        ks = calibrated_k(clean, k, 20)
        d = [first_outside_after(m[k], mu, sd, ks, 10) for m in ons]
        print(f"{cond:<16} {k:<15} delays="
              f"{[None if x is None else x - 10 for x in d]}")

print("\n== onset losses, epoch 10 -> 11 (mean of 5 runs) ==")
for cond in PAIRS:
    ons = load_runs("results", f"{cond}_onset10")
    tl = np.array([m["train_loss"] for m in ons])
    te = np.array([m["test_loss"] for m in ons])
    print(f"{cond:<16} train_loss {tl[:, 9].mean():.3f} -> {tl[:, 10].mean():.3f}   "
          f"test_loss {te[:, 9].mean():.3f} -> {te[:, 10].mean():.3f}")

print("\n== retracted v1 first_change on failure runs (0-based index) ==")
for cond in PAIRS:
    runs = load_runs("results", cond)
    out = {k: [first_change(m[k]) for m in runs] for k in METRICS}
    print(f"{cond:<16}",
          {k: f"{np.mean(v):.1f}+/-{np.std(v):.2f}" for k, v in out.items()})


def du(p):
    return sum(os.path.getsize(os.path.join(d, f))
               for d, _, fs in os.walk(p) for f in fs)


print("\n== storage ==")
print(f"results/ {du('results') / 1024:.0f} KiB   figures/ {du('figures') / 2**20:.1f} MiB   "
      f"paper PDF {os.path.getsize('Silent_Failure_Detection.pdf') / 2**20:.2f} MiB")
print(subprocess.run(["git", "count-objects", "-vH"],
                     capture_output=True, text=True).stdout)
