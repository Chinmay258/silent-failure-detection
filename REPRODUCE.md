# Reproducing this project from scratch

Everything needed to rebuild every number, table and figure in this repository.
Nothing here depends on files that live outside the repository: the datasets
download themselves, and the model weights are never needed because every
result is derived from the per-epoch metrics in `results/`.

---

## 1. What is committed, and what is regenerated

| Artifact | In the repo? | How to rebuild |
|---|---|---|
| Source code (9 modules + `analysis/`) | yes | — |
| Per-epoch metrics for all 80 runs (`results/`) | yes, ~340 KB | `python run.py` and the commands in §5 (~16 GPU-hours) |
| Figures (`figures/`) | yes, 6.5 MB | `python plot_ref.py`, `plot_onset.py`, `plot.py` (seconds) |
| v1 preprint PDF | yes | superseded — see "Known issues" in `README.md` |
| MNIST / CIFAR-10 (`data/`, 404 MB) | **no** (gitignored) | downloads automatically on first run |
| Virtualenv (`.venv/`, 4.6 GB) | **no** (gitignored) | §3 |
| Model weights / checkpoints | **no**, never saved | not required; see §7 |

Every analysis table reproduces from the committed metrics in seconds, with no
retraining and no GPU.

---

## 2. Hardware and software the results came from

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU, 4.0 GiB VRAM, compute capability 8.6 |
| CUDA / cuDNN | build 12.1 / 9.1.0 |
| OS | Windows 11 (10.0.26300) |
| Python | 3.11.9 |
| Key packages | torch 2.5.1+cu121, torchvision 0.20.1+cu121, numpy 2.4.4, matplotlib 3.10.8 |

The 4 GiB VRAM limit is the binding constraint on batch size: peak allocation
is 299 MiB for ResNet-18 at batch 256, so there is headroom, but nothing here
was tuned for larger cards.

---

## 3. Environment setup

```bash
git clone https://github.com/Chinmay258/silent-failure-detection.git
cd silent-failure-detection
python -m venv .venv
# Windows:
.venv/Scripts/activate
# Linux/macOS:
# source .venv/bin/activate

# GPU (exact versions used):
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-lock.txt

# CPU-only, or any newer stack:
# pip install -r requirements.txt
```

`requirements.txt` holds loose lower bounds and is what a fresh reader should
use; `requirements-lock.txt` pins the exact versions behind the committed
numbers.

---

## 4. Datasets

Both are fetched by `torchvision` into `data/` on first use and are **not**
committed (404 MB). Every dataset construction re-verifies the MD5 of each
archive, so a partial or corrupted download fails loudly rather than silently
training on bad data.

| Dataset | Source | Used by |
|---|---|---|
| MNIST | `torchvision.datasets.MNIST` | `label_noise`, `clean_mnist` |
| CIFAR-10 | `torchvision.datasets.CIFAR10` | the other six conditions |

Deleting `data/` is safe at any time; the next run re-downloads it.

---

## 5. Regenerating all 80 runs

Each `train.py` invocation writes `results/<condition>/run<N>/metrics.npy`, one
file per run, written once at the end of the run. There is no checkpointing: a
run interrupted at epoch 19 leaves nothing behind and must be repeated. Runs
are independent, so a partially completed sweep can be resumed by rerunning
only the missing run indices.

**Four failure conditions** (5 seeds each):

```bash
python run.py                       # trains all four, then evaluates and plots
# or individually:
python train.py --condition label_noise     --epochs 20 --runs 5
python train.py --condition spurious        --epochs 20 --runs 5
python train.py --condition dist_shift      --epochs 20 --runs 5
python train.py --condition class_imbalance --epochs 20 --runs 5
```

**Four clean-reference conditions** (10 seeds each) — these define the
detection baseline and must exist before any `evaluate_ref` / `evaluate_onset`
analysis:

```bash
python train.py --condition clean_mnist     --epochs 20 --runs 10
python train.py --condition clean_cifar_raw --epochs 20 --runs 10
python train.py --condition clean_cifar     --epochs 20 --runs 10
python train.py --condition clean_resnet    --epochs 20 --runs 10
```

**Mid-training onset variants** (5 seeds each) — clean for 10 epochs, then the
corruption switches on:

```bash
python train.py --condition label_noise     --epochs 20 --runs 5 --onset 10
python train.py --condition spurious        --epochs 20 --runs 5 --onset 10
python train.py --condition dist_shift      --epochs 20 --runs 5 --onset 10
python train.py --condition class_imbalance --epochs 20 --runs 5 --onset 10
```

**Same-stream study** (3 seeds, measures every signal on both the clean and the
corrupted stream — the comparison `train.py` does not make):

```bash
python analysis/fair_baseline_v2.py                # all three seeds
python analysis/fair_baseline_v2.py --seeds 42     # resume a single seed
python analysis/fair_baseline_v2.py --summary-only # re-print the summary
```

Seeds already on disk are skipped, and the summary is computed from whatever
seed files exist, so a partial sweep still reports. Run seeds one at a time on
a memory-tight machine: `effective_rank` takes a full float64 SVD of a
10,000 × 512 matrix and keeps the singular vectors, and running all three seeds
in one process accumulates that. This sweep was in fact interrupted once by a
host-RAM allocation failure inside `np.linalg.svd` — note that this is *system*
RAM, not VRAM, and that a nearly full disk makes it worse because Windows
cannot grow the pagefile.

### Measured cost

| Model / data | Per epoch | Fixed per-run overhead | Peak GPU |
|---|---|---|---|
| SimpleCNN / MNIST | 9.6 s | 16.1 s (includes first CUDA init) | 133 MiB |
| SimpleCNN / CIFAR-10 | 44.0 s | 1.3 s | 171 MiB |
| ResNet-18 / CIFAR-10 | 48.8 s | 3.1 s | 299 MiB |

Full sweep ≈ **16 GPU-hours** on the hardware above (400 MNIST-CNN epochs,
800 CIFAR-CNN epochs, 400 ResNet-18 epochs).

Most of a CIFAR-10 epoch is **not** GPU work. `train.py` uses
`num_workers=4` without `persistent_workers`, and iterates three loaders per
epoch, so twelve worker processes are started per epoch. Each re-imports torch
(~3.4 s) and receives the dataset by pickle — torchvision keeps CIFAR-10's
images as a NumPy array, which is copied by value into every worker (29.3 MiB
for the test split, 146.6 MiB for train), whereas MNIST's tensor is passed
through shared memory. Measured: ~11.4 s before the first CIFAR batch versus
3.3 s for MNIST, against 0.33 s for the actual GPU forward pass over 10,000
images. Setting `persistent_workers=True` drops later passes to 0.4 s.

### Determinism

Seeds are fixed (`seed = run * 42`, set for `torch`, `numpy` and `random`), but
runs are **not** bit-for-bit reproducible: cuDNN selects non-deterministic
kernels and no deterministic flags are set. Re-running a seed reproduces the
numbers to within normal run-to-run variation, not exactly. Two of the
committed `dist_shift_onset10` runs were in fact regenerated after an
interruption and are not bit-identical to a single uninterrupted sweep.

---

## 6. Regenerating every analysis and figure

These need only the committed `results/` — no GPU, no datasets, seconds each:

```bash
python evaluate_ref.py   --results results              # calibrated reference-band detection
python evaluate_onset.py --results results --onset 10   # detection delay after onset
python evaluate.py       --results results              # the retracted v1 metric (kept as an ablation)

python plot_ref.py   --results results --out figures            # clean vs failure, per condition
python plot_onset.py --results results --out figures --onset 10 # signals and baselines around onset
python plot.py       --results results --out figures            # the v1 2x2 normalized-trajectory figure

python analysis/card_numbers.py     # every headline figure, recomputed
python analysis/verify_claims.py    # the verification suite (needs data/ for §6, §10)
```

---

## 7. Design choices that affect reproduction

- **No checkpoints.** Metrics are written once, at the end of a run. This keeps
  `results/` tiny and makes a results file mean "this run completed", but a
  crash loses the whole run. The runs are short enough that this was the right
  trade; a longer sweep should checkpoint.
- **No model weights are saved.** Every claim is derived from per-epoch scalar
  metrics, so the weights are not needed to reproduce any table. Re-deriving a
  *new* metric from old runs does require retraining.
- **One file per run, not a database.** 80 small `.npy` files version cleanly in
  git alongside the code that produced them. They do not record which commit
  produced them — at this scale the commit history serves that purpose.
- **Analysis is separate from training.** `train.py` only writes; the
  `evaluate*`/`plot*` modules only read. The detection method could therefore
  be rebuilt and re-evaluated without touching the 16 GPU-hours of runs — which
  is exactly what happened when the original detection metric was retracted.

---

## 8. Things that will trip up a re-run

- **`load_runs` stops at the first gap.** It probes `run0`, `run1`, … and stops
  at the first missing index, so a results directory missing `run3` silently
  yields 3 runs instead of 9. Verify counts after a partial sweep.
- **A single NaN disables detection silently.** NaN propagates into the
  calibrated threshold, every comparison becomes false, and the detector
  reports "no detections" rather than failing. Demonstrated in
  `analysis/verify_claims.py` §8.
- **Reference runs are required first.** `evaluate_ref.py` and
  `evaluate_onset.py` need at least 3 (and are calibrated on 10) runs of the
  matching `clean_*` condition; they skip conditions whose reference is absent.
- **Five runs predate the expanded metric set.** `runs 0–4` of each `clean_*`
  condition store 4 metrics; runs 5–9 and all onset runs store 10. Analyses
  that use the baseline signals must restrict themselves accordingly.
- **The label-noise rate is ~27%, not 30%.** Indices are chosen for 30% of the
  training set, but the replacement label is drawn uniformly over all classes
  and can coincide with the original.
- **Two classes share a shortcut colour.** The spurious-feature hue map sends
  class 0 and class 9 to the same red patch, so the shortcut is ambiguous for
  that pair.
- **`dist_shift` is a harsher corruption than "σ = 0.1 noise".**
  `add_gaussian_noise` applies `.clamp(0, 1)` to *normalized* inputs (range
  ≈ −2 to 2.1), zeroing ~52% of every image. See "Known issues" in `README.md`.
