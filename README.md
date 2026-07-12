# Silent Failure Detection in Deep Neural Networks

**Penultimate-layer embedding geometry — dispersion and effective rank — as a training-time monitoring signal.**

Chinmay Singh, Amreen Ayesha · Manipal Institute of Technology Bengaluru (MAHE)

📄 Paper: [`Silent_Failure_Detection.pdf`](./Silent_Failure_Detection.pdf) *(in this repo)*

---

## Overview

A neural network can fail **silently**: its accuracy and softmax confidence stay high while its internal representations quietly degrade. By the time output metrics react, the damage is already done. This project asks a simple question:

> Can the **geometry of learned representations** warn us that something is wrong *before* accuracy does?

The answer we found is **"sometimes" — and we report exactly when it does and when it doesn't.**

We track two scalar metrics on the penultimate-layer embeddings after every training epoch. Both are **training-free** and **gradient-free** — they require no change to the training procedure and no access to gradients:

- **Embedding dispersion** — the average distance of embeddings from their centroid (how spread out the representations are).
- **Effective rank** — `exp(` Shannon entropy of the normalized singular values `)` — how many dimensions meaningfully contribute to the representation.

## Failure conditions studied

| Condition | Dataset | Architecture | What it simulates |
|---|---|---|---|
| Label noise | MNIST | CNN | 30% of training labels randomly corrupted |
| Spurious features | CIFAR-10 | CNN | A class-correlated colour patch present in 90% of train images, removed at test |
| Distribution shift | CIFAR-10 | ResNet-18 | Gaussian noise (σ = 0.1) added at evaluation |
| Class imbalance | CIFAR-10 | CNN | Two classes make up 90% of training data |

All models trained for 20 epochs (Adam, lr 1e-3, batch 256), **5 independent seeds per condition**, no early stopping, no geometry-aware regularization.

## Results

First-change epoch (mean ± std over 5 runs). Lower = reacted sooner. **Δ** = geometry's lead over accuracy in epochs; positive means geometry fired first.

| Condition | Accuracy | Confidence | Dispersion | Effective Rank | ΔDisp | ΔERank |
|---|---|---|---|---|---|---|
| Label Noise (MNIST) | 10.4 ± 0.49 | 1.2 ± 0.40 | 7.2 ± 1.94 | 1.8 ± 0.40 | **+3.2** | **+8.6** |
| Spurious Features (CIFAR-10) | 1.2 ± 0.40 | 1.6 ± 0.80 | 1.0 ± 0.00 | 2.6 ± 0.49 | +0.2 | −1.4 |
| Distribution Shift (ResNet-18) | 1.0 ± 0.00 | 1.0 ± 0.00 | 1.4 ± 0.80 | 1.0 ± 0.00 | −0.4 | 0.0 |
| Class Imbalance (CIFAR-10) | 1.0 ± 0.00 | 1.2 ± 0.40 | 2.6 ± 0.49 | 1.0 ± 0.00 | −1.6 | 0.0 |

### Key findings

- **Label noise is where geometry shines.** Effective rank crossed the detection threshold ~8.6 epochs before accuracy did — a large, consistent early-warning window.
- **Confidence can actively mislead.** Under spurious features, softmax confidence kept *rising* throughout training even as the model grew more dependent on a colour patch that vanishes at test time.
- **Honest nulls.** Under distribution shift and class imbalance, every signal fired at epoch 1 — geometry offered no timing advantage. We report this as directly as the positive results.
- **Effective rank > dispersion** as a monitoring signal: lower run-to-run variance in three of the four conditions (dispersion was the steadier signal under spurious features).

**Takeaway:** geometry monitoring is not a universal upgrade. Its value is specifically tied to whether representational degradation is *gradual* (where effective rank gives real lead time) or *abrupt* (where nothing beats accuracy).

## Method notes

- **Lead-time metric.** For a normalized signal `s(t)`, the first-change epoch is the smallest `t > 0` with `|s(t) − s(0)| > τ`, where `τ = 0.1`, fixed across all conditions. Lead `Δ = t_accuracy − t_geometry`.
- No metric requires gradient access or modifies training; both are computed from the embedding matrix `Z ∈ ℝ^(N×d)` at each epoch.

## Repository structure

```
.
├── README.md
├── Silent_Failure_Detection.pdf   # the paper (preprint)
├── requirements.txt               # torch, torchvision, numpy, matplotlib
├── run.py                         # single entry point: train → evaluate → plot
├── train.py                       # training engine, models, and the 4 failure conditions
├── geometry.py                    # dispersion() and effective_rank() metrics
├── evaluate.py                    # lead-time analysis; prints the results table
├── plot.py                        # renders the 2×2 signal-trajectory figure
├── results/                       # saved per-run metrics backing the paper
│   ├── label_noise/run{0..4}/metrics.npy
│   ├── spurious/run{0..4}/metrics.npy
│   ├── dist_shift/run{0..4}/metrics.npy
│   └── class_imbalance/run{0..4}/metrics.npy
└── figures/
    └── main_results.png           # the headline 2×2 figure
```

Datasets (MNIST / CIFAR-10) download on demand into `data/` and are **not** committed.

## Installation

Requires Python 3.11+. A CUDA-capable GPU is recommended for training but not
required (the analysis/plotting scripts run on CPU).

```bash
git clone https://github.com/Chinmay258/silent-failure-detection.git
cd silent-failure-detection
python -m venv .venv && source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

`run.py` is the single entry point — it trains, evaluates, and plots in one shot.
Datasets are downloaded automatically on first run.

```bash
# Full pipeline: all 4 conditions, 5 seeds × 20 epochs (defaults)
python run.py

# Custom epochs / seeds
python run.py --epochs 15 --runs 3

# A single condition (label_noise | spurious | dist_shift | class_imbalance)
python run.py --condition label_noise
```

The repo ships with the saved `results/` from the paper, so you can reproduce the
table and figure **without retraining**:

```bash
python run.py --skip-train          # evaluate + plot from existing results/
# or run the stages directly:
python evaluate.py --results results          # prints the lead-time table
python plot.py --results results --out figures # writes figures/main_results.png
```

Individual training runs can also be launched directly:

```bash
python train.py --condition label_noise --epochs 20 --runs 5 --out results
```

## Limitations

- Experiments are limited to image classification on MNIST and CIFAR-10, chosen for fast, reproducible training; behaviour at larger scale (bigger models, longer runs) remains open.
- The threshold `τ = 0.1` was fixed for convenience; principled threshold selection is future work.
- These metrics are a **monitoring** tool, not a **diagnostic** one — they tell you *something* changed, not *what* or *why*.

## Citation

```bibtex
@misc{singh_silent_failure_2026,
  title  = {Silent Failure Detection in Deep Neural Networks using Penultimate-Layer
            Embedding Geometry: Dispersion and Effective Rank as Training-Time Monitoring Signals},
  author = {Singh, Chinmay and Ayesha, Amreen},
  year   = {2026},
  note   = {Preprint}
}
```
