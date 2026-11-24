# Silent Failure Detection in Deep Neural Networks

**Label-free detection of silent model failure from penultimate-layer embedding geometry.**

Chinmay Singh · Manipal Institute of Technology Bengaluru (MAHE)

---

## Overview

A model can fail **silently**: validation accuracy and softmax confidence look healthy while the
model served in production is badly degraded. This project asks whether the **geometry of a
model's internal representations**, computed on *unlabeled* incoming data, can catch that.

The headline result, on CIFAR-10 / ResNet-18 under input drift:

| Measured on | Accuracy |
|---|---|
| Clean validation stream | **74.2%** |
| The drifted stream the model actually serves | **39.9%** |

A **34-point collapse that validation accuracy cannot see at all.** Embedding dispersion,
computed without labels on the incoming stream, separates from the clean reference **2.8×**
more strongly than the standard softmax-confidence OOD baseline.

## This project corrected itself — read this part

The first version of this work reported that effective rank detected label-noise failure
**8.6 epochs before accuracy**. That result does not survive its own control, and the repo now
documents why.

The original detector flagged failure at the first epoch a normalized signal moved more than
τ = 0.1 from its starting value. Nobody had asked what that detector does on a **healthy**
training run. It fires there too — within the first two epochs, on every signal, across all four
clean configurations (40 control runs). "Moved early" never meant "detected something"; it
measured ordinary warm-up dynamics. The normalization also used the whole run's min/max, so the
rule needed information from epoch 20 to fire at epoch 2 — not computable online.

**What replaced it.** Detection is now measured against **clean-reference bands**: train clean
control runs with an identical pipeline, form a per-epoch mean/std band, and flag a monitored run
when its z-score exceeds a threshold `k*` calibrated as the worst leave-one-out excursion the
clean runs themselves make — the smallest threshold with zero false alarms on the reference set.
Every detection is reported alongside its false-alarm rate. Out-of-sample: **1 false alarm across
200 held-out clean epochs** (from effective rank; dispersion: zero).

Under that protocol, two of the original conclusions inverted:

- **Dispersion, not effective rank, is the dependable signal.** Effective rank's separation
  *decays* over training (22.6% at epoch 1 → 2.2% by epoch 20) and it produced the only
  out-of-sample false alarm. The original "effective rank is more reliable" claim is retracted.
- **Most of these failures are not silent.** Once any clean reference exists, label noise,
  spurious correlations, and class imbalance are all caught at least as fast by accuracy or by the
  training loss. Geometry's real value is not speed — it is **coverage** of the one failure class
  output metrics structurally cannot see: drift in the unlabeled input stream.

## Failure conditions studied

| Condition | Dataset | Architecture | What it simulates |
|---|---|---|---|
| Label noise | MNIST | CNN | 30% of training labels randomly corrupted |
| Spurious features | CIFAR-10 | CNN | Class-correlated colour patch in 90% of train images, removed at test |
| Distribution shift | CIFAR-10 | ResNet-18 | Input corruption applied to the monitored stream |
| Class imbalance | CIFAR-10 | CNN | Two classes make up 90% of training data |

Each condition has a **clean-training control** with an identical data pipeline and architecture
(10 seeds), and a **mid-training onset** variant where the corruption switches on after epoch 10
so detection delay is measured against a known onset. 20 epochs, Adam lr 1e-3, batch 256.
**80+ training runs total**, all committed.

## Results

Calibrated reference-band detection (thresholds with zero false alarms on the reference set):

| Condition | From scratch | Onset delay (corruption at epoch 10) |
|---|---|---|
| Label noise | Conf/Disp: ep 1 · ERank: 2.4 · Acc: 6.4 | Conf/Disp: 1 ep · Acc: 1.8 · train loss ×200 |
| Spurious features | Acc: ep 1 (3.0×) · ERank: ep 1 (1.2×) | Conf: 8.0 · Disp: 7.4 · Acc/ERank: never |
| Class imbalance | Acc: ep 1 (6.9×) | none within 10 epochs (train loss *improves*) |
| **Distribution shift** | **Disp: ep 2 (5.6×) · Acc/Conf: never** | **Disp: 1.0 ± 0.0 · Acc/Conf: never** |

Same-stream comparison under drift (all signals computed on the *same* unlabeled drifted data,
3 seeds):

| Signal | Relative separation from clean |
|---|---|
| Embedding dispersion | **37.6%** |
| Softmax confidence (Hendrycks & Gimpel baseline) | 13.2% |
| Effective rank | 22.6% at epoch 1 → 2.2% by epoch 20 |

## Known issues / work in progress

This is an honest snapshot of a project mid-revision, not a finished paper.

- **The `dist_shift` corruption is stronger than its description.** `add_gaussian_noise` applies
  `.clamp(0, 1)` to *normalized* inputs (range ≈ [-2, 2.1]), so it zeroes ~52% of every image
  rather than adding mild σ = 0.1 noise. The blindness of validation accuracy to it is real, but
  the corruption must be re-implemented and re-described before publication.
- **A model-free baseline also works on this corruption.** Per-image pixel standard deviation
  separates clean from corrupted data at AUROC 0.986 with no model at all
  (`analysis/verify_claims.py` §10), so this corruption does not establish that a model-aware
  signal is needed. A benchmark where that is not true (e.g. CIFAR-10-C) is the right next
  experiment.
- **"Only geometry reacts" does not survive a like-for-like comparison.** Measured on the *same*
  corrupted stream (`analysis/fair_baseline_v2.py`), max softmax probability separates too.
  Mean dispersion shifts ~3× more than mean confidence, but per-sample AUROC is 0.76 against
  0.70. What does hold is that validation accuracy is blind: ~74% while true accuracy on the
  corrupted stream is ~39%.
- Mahalanobis and energy-based OOD baselines are cited but not yet compared against.
- Failure conditions use 5 seeds against 10-seed references.
- The paper PDF in this repo is the **superseded v1** and still contains the retracted claim.

## Repository structure

```
.
├── README.md
├── Silent_Failure_Detection.pdf   # v1 preprint (superseded — see Known issues)
├── requirements.txt
├── run.py                         # entry point: train → evaluate → plot
├── train.py                       # training engine, models, failure + clean-control conditions
├── geometry.py                    # dispersion, effective rank, class-conditional variants
├── evaluate.py                    # v1 first-change analysis (kept as a cautionary ablation)
├── evaluate_ref.py                # reference-band detection + calibrated zero-false-alarm k*
├── evaluate_onset.py              # detection delay after a mid-training onset
├── plot.py / plot_ref.py / plot_onset.py
├── analysis/                      # verification suite and supporting studies (see analysis/README.md)
│   ├── fair_baseline_v2.py        # measures every signal on the SAME corrupted stream
│   ├── verify_claims.py           # 12 checks of this project's claims against its own runs
│   ├── card_numbers.py            # recomputes every headline figure from results/
│   ├── bench_geometry.py          # scaling cost of the geometry metrics
│   ├── time_epoch.py              # real per-epoch wall clock and peak GPU memory
│   └── rng_workers_demo.py        # DataLoader worker RNG bug in the shortcut condition
├── REPRODUCE.md                   # full rebuild instructions, hardware, timings, pitfalls
├── requirements-lock.txt          # exact pinned versions behind the committed numbers
├── results/                       # 80 runs: failures, clean controls, onset variants
└── figures/
```

## Installation

Requires Python 3.11+. A CUDA-capable GPU is recommended for training; the analysis and plotting
scripts run on CPU.

```bash
git clone https://github.com/Chinmay258/silent-failure-detection.git
cd silent-failure-detection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Full rebuild instructions — hardware, exact versions, every command, measured
timings and the pitfalls — are in **[REPRODUCE.md](REPRODUCE.md)**.

The repo ships with all saved runs, so every table above reproduces **without retraining**:

```bash
python evaluate_ref.py --results results
```

```bash
python evaluate_onset.py --results results --onset 10
```

```bash
python plot_ref.py --results results --out figures
```

To retrain from scratch (datasets download on demand):

```bash
python run.py
```

```bash
python train.py --condition clean_resnet --epochs 20 --runs 10
```

```bash
python train.py --condition dist_shift --epochs 20 --runs 5 --onset 10
```

## Limitations

- Image classification on MNIST and CIFAR-10 only; behaviour at larger scale is open.
- Detection requires clean reference runs. Reference-free online change-point detection is future
  work.
- These metrics are a **monitoring** tool, not a **diagnostic** one — they tell you *something*
  changed, not *what* or *why*.

## Citation

```bibtex
@misc{singh_silent_failure_2026,
  title  = {Silent Failure Under Input Drift: Label-Free Detection Using
            Penultimate-Layer Embedding Dispersion},
  author = {Singh, Chinmay},
  year   = {2026},
  note   = {Preprint in preparation},
  url    = {https://github.com/Chinmay258/silent-failure-detection}
}
```

## Acknowledgments

Thanks to Prof. Amreen Ayesha for supporting this project.
