"""
Verification suite: checks the project's own claims against the committed runs
and exercises the code paths where the implementation and the description
disagree. Every number quoted in the README and in the paper revision should
be reproducible from this script plus `results/`.

Each section prints the file:line it is checking so a reader can follow along.

Usage:
    python analysis/verify_claims.py
"""

import copy
import inspect
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)

import numpy as np
import torch
import torchvision

from evaluate import load_runs, first_change, METRICS
from evaluate_ref import PAIRS, band, detect, calibrated_k
from train import add_gaussian_noise


def hdr(s):
    print("\n" + "=" * 78 + "\n" + s + "\n" + "=" * 78)


# ---------------------------------------------------------------------------
hdr("1. RUN INVENTORY AND METRIC SCHEMA  (train.py: metrics dict; np.save at end)")
total = 0
for cond in sorted(os.listdir("results")):
    cdir = Path("results") / cond
    if not cdir.is_dir():
        continue
    runs = sorted((r for r in os.listdir(cdir) if r.startswith("run")),
                  key=lambda r: int(r[3:]))
    if not runs:
        continue
    sch = []
    for r in runs:
        m = np.load(cdir / r / "metrics.npy", allow_pickle=True).item()
        sch.append((r, len(m), len(m["accuracy"])))
    total += len(runs)
    keysets = sorted({s[1] for s in sch})
    print(f"{cond:<26} runs={len(runs):>2}  keys/run={keysets}  "
          f"epochs={sorted({s[2] for s in sch})}")
    if len(keysets) > 1:
        print("    MIXED SCHEMA (runs predating the expanded metric set):",
              [(r, k) for r, k, _ in sch])
print("TOTAL metrics.npy files:", total)

# ---------------------------------------------------------------------------
hdr("2. dist_shift VALIDATION ACCURACY  (train.py eval loop, clean test set)")
ds, cr = load_runs("results", "dist_shift"), load_runs("results", "clean_resnet")
a = np.array([m["accuracy"][-1] for m in ds])
c = np.array([m["accuracy"][-1] for m in cr])
print(f"dist_shift   final clean-test acc: {a.mean():.4f} +/- {a.std():.4f}  (n={len(a)})")
print(f"clean_resnet final acc:            {c.mean():.4f} +/- {c.std():.4f}  (n={len(c)})")
print("metric keys stored for dist_shift:", sorted(ds[0].keys()))
print("-> accuracy on the CORRUPTED stream is not stored by train.py;")
print("   analysis/fair_baseline_v2.py measures it.")

# ---------------------------------------------------------------------------
hdr("3. THE v1 CONFOUND: first_change() on CLEAN runs  (evaluate.py:first_change)")
print("note: first_change returns a 0-based INDEX; the smallest value it can "
      "return is 1 (the 2nd epoch)")
for ref in ["clean_mnist", "clean_cifar_raw", "clean_cifar", "clean_resnet"]:
    runs = load_runs("results", ref)
    out = {k: [first_change(m[k]) for m in runs] for k in METRICS}
    print(f"{ref:<16} " + "  ".join(f"{k}: mean={np.mean(v):.1f} max={max(v)}"
                                    for k, v in out.items()))
print("-> every signal 'fires' on healthy training, so the v1 lead-time claim "
      "measured warm-up, not failure.")

# ---------------------------------------------------------------------------
hdr("4. OUT-OF-SAMPLE FALSE ALARMS  (pre-onset epochs of the onset runs)")
runs_fired = epoch_cross = clean_epochs = pairs = 0
details = []
for cond, ref in PAIRS.items():
    onset, clean = load_runs("results", f"{cond}_onset10"), load_runs("results", ref)
    if not onset or len(clean) < 3:
        continue
    H = min(len(m["accuracy"]) for m in onset + clean)
    clean_epochs += len(onset) * 10
    for key in METRICS:
        mu, sd = band(clean, key)
        ks = calibrated_k(clean, key, H)
        for i, m in enumerate(onset):
            pairs += 1
            x = np.asarray(m[key][:10], dtype=float)
            cross = np.abs(x - mu[:10]) > ks * sd[:10]
            n = int(cross.sum())
            epoch_cross += n
            if n:
                runs_fired += 1
                details.append((cond, key, f"run{i}",
                                (np.where(cross)[0] + 1).tolist(), round(ks, 2)))
print(f"held-out clean epochs (conditions x runs x 10): {clean_epochs}")
print(f"run-signal pairs evaluated: {pairs};  pairs with >=1 crossing: {runs_fired}")
print(f"epoch-signal checks: {clean_epochs * 4};  crossings: {epoch_cross}")
for d in details:
    print("   crossing:", d)

# ---------------------------------------------------------------------------
hdr("5. HOW HELD-OUT ARE THEY? onset runs reuse reference seeds (seed = run * 42)")
for cond, ref in PAIRS.items():
    onset, clean = load_runs("results", f"{cond}_onset10"), load_runs("results", ref)
    if not onset or not clean:
        continue
    for key in ["accuracy", "dispersion"]:
        same, other = [], []
        for i, m in enumerate(onset):
            x = np.asarray(m[key][:10])
            same.append(np.mean(np.abs(x - np.asarray(clean[i][key][:10]))))
            other.append(np.mean([np.mean(np.abs(x - np.asarray(clean[j][key][:10])))
                                  for j in range(len(clean)) if j != i]))
        print(f"{cond:<16} {key:<11} |onset_i - ref_i| = {np.mean(same):.4f}   "
              f"|onset_i - ref_j| = {np.mean(other):.4f}")
print("-> same-index pairs are measurably closer: the onset runs are not fully "
      "independent of the reference runs.")

# ---------------------------------------------------------------------------
hdr("6. EFFECTIVE LABEL-NOISE RATE  (train.py NoisyLabelDataset)")
mn = torchvision.datasets.MNIST("data", train=True, download=True)
t = mn.targets.numpy()
for seed in range(5):
    rng = np.random.default_rng(seed)
    lab = t.copy()
    flip = rng.choice(len(t), size=int(0.3 * len(t)), replace=False)
    lab[flip] = rng.integers(0, int(t.max()) + 1, size=len(flip))
    print(f"seed {seed}: indices selected {len(flip) / len(t):.3f}   "
          f"labels actually changed {(lab != t).mean():.4f}")
print("-> the redraw can pick the original label, so 30% selected is ~27% changed.")

# ---------------------------------------------------------------------------
hdr("7. SPURIOUS PATCH COLOURS  (train.py SpuriousDataset hue mapping)")
cols = []
for label, h in enumerate(np.linspace(0, 1, 10)):
    rgb = tuple(np.clip([abs(h * 6 - 3) - 1, 2 - abs(h * 6 - 2),
                         2 - abs(h * 6 - 4)], 0, 1).round(3))
    cols.append(rgb)
    print(f"class {label}: hue={h:.3f} rgb={rgb}")
dupes = [(i, j) for i in range(10) for j in range(i + 1, 10) if cols[i] == cols[j]]
print("classes sharing an identical patch:", dupes,
      "-> hue 0.0 and 1.0 are both red, so the shortcut is ambiguous for that pair.")

# ---------------------------------------------------------------------------
hdr("8. ONE NaN DISABLES DETECTION SILENTLY  (evaluate_ref.py band/calibrated_k)")
clean, fail = load_runs("results", "clean_resnet"), load_runs("results", "dist_shift")
mu0, sd0 = band(clean, "dispersion")
k0 = calibrated_k(clean, "dispersion", 20)
print(f"healthy refs       : k*={k0:.2f}  detections="
      f"{[detect(m['dispersion'], mu0, sd0, k0) for m in fail]}")
bad = copy.deepcopy(clean)
bad[3]["dispersion"][7] = float("nan")
mu1, sd1 = band(bad, "dispersion")
k1 = calibrated_k(bad, "dispersion", 20)
print(f"1 NaN, run3 epoch 8: k*={k1}  detections="
      f"{[detect(m['dispersion'], mu1, sd1, k1) for m in fail]}")
print("-> NaN propagates into k*, every comparison becomes False, and the "
      "detector reports 'no detections' instead of failing.")

# ---------------------------------------------------------------------------
hdr("9. load_runs() STOPS AT THE FIRST GAP  (evaluate.py:load_runs)")
tmp = tempfile.mkdtemp()
try:
    for r in [0, 1, 2, 4, 5, 6, 7, 8, 9]:
        d = Path(tmp) / "clean_resnet" / f"run{r}"
        d.mkdir(parents=True)
        shutil.copy(Path("results") / "clean_resnet" / f"run{r}" / "metrics.npy", d)
    got = len(load_runs(tmp, "clean_resnet"))
    print(f"on disk: 9 runs (run3 missing) -> load_runs returned {got}, no warning")
finally:
    shutil.rmtree(tmp)

# ---------------------------------------------------------------------------
hdr("10. WHAT dist_shift ACTUALLY DOES + A MODEL-FREE BASELINE  (train.py:add_gaussian_noise)")
cf = torchvision.datasets.CIFAR10("data", train=False, download=True)
x = torch.from_numpy(cf.data).float().permute(0, 3, 1, 2) / 255.0
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
xn = (x - MEAN) / STD
torch.manual_seed(0)
y = add_gaussian_noise(xn, std=0.1)
print(f"normalized input range [{xn.min():.2f}, {xn.max():.2f}] -> "
      f"output range [{y.min():.2f}, {y.max():.2f}]")
print(f"pixels < 0 before clamp (zeroed): {(xn < 0).float().mean() * 100:.1f}%   "
      f"pixels > 1 (clipped): {(xn > 1).float().mean() * 100:.1f}%")
print(f"mean |pixel change|: {(y - xn).abs().mean():.3f}   "
      f"(pure N(0,0.1) noise would be {0.1 * np.sqrt(2 / np.pi):.3f})")
sc, sn = xn.std(dim=(1, 2, 3)).numpy(), y.std(dim=(1, 2, 3)).numpy()
s = np.r_[sc, sn]
lab = np.r_[np.ones(len(sc)), np.zeros(len(sn))]
ranks = np.empty(len(s))
ranks[s.argsort()] = np.arange(1, len(s) + 1)
auc = (ranks[lab == 1].sum() - len(sc) * (len(sc) + 1) / 2) / (len(sc) * len(sn))
print(f"per-image pixel std: clean {sc.mean():.3f}+/-{sc.std():.3f}   "
      f"corrupted {sn.mean():.3f}+/-{sn.std():.3f}")
print(f"model-free 'per-image std' detector AUROC: {auc:.4f}")
print("-> the clamp is written for [0,1] images but receives normalized ones, so "
      "this is a severe corruption, and no model is needed to spot it.")

# ---------------------------------------------------------------------------
hdr("11. SAME-STREAM STUDY  (analysis/fair_baseline_v2.py)")
files = sorted((REPO / "results" / "fair_baseline_v2").glob("seed*.npy"))
if not files:
    print("not present - run: python analysis/fair_baseline_v2.py")
else:
    ms = [np.load(f, allow_pickle=True).item() for f in files]
    g = lambda k: np.array([m[k] for m in ms])
    ac, an = g("acc_clean"), g("acc_noised")
    print(f"files: {[f.name for f in files]}")
    print(f"acc clean     ep20 {ac[:, -1].mean():.3f} +/- {ac[:, -1].std():.3f}")
    print(f"acc corrupted ep20 {an[:, -1].mean():.3f} +/- {an[:, -1].std():.3f}")
    gap = 100 * (ac[:, -1] - an[:, -1])
    print(f"hidden gap: {gap.mean():.1f} +/- {gap.std():.1f} points")
    rel = {}
    for name in ("conf", "disp", "erank"):
        cl, no = g(f"{name}_clean"), g(f"{name}_noised")
        rel[name] = (np.abs(no - cl) / np.abs(cl)).mean(axis=1)
        print(f"{name:<6} relative separation {100 * rel[name].mean():5.1f}%")
    print(f"dispersion/confidence ratio: {(rel['disp'] / rel['conf']).mean():.2f}")
    for key in ("auroc_msp", "auroc_centroid_dist"):
        for ep in (1, 20):
            v = np.array([m[key][ep] for m in ms])
            print(f"{key:<20} epoch {ep:>2}: AUROC {v.mean():.4f} "
                  f"(two-sided {np.maximum(v, 1 - v).mean():.4f})")

# ---------------------------------------------------------------------------
hdr("12. DATASETS ARE INTEGRITY-CHECKED ON EVERY CONSTRUCTION  (torchvision)")
import torchvision.datasets.cifar as cifm
print("CIFAR10.__init__ calls self.download():",
      "self.download()" in inspect.getsource(cifm.CIFAR10.__init__))
print("CIFAR10.download() checks MD5 of every batch file first:",
      "_check_integrity" in inspect.getsource(cifm.CIFAR10.download))
