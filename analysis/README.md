# Analysis and verification scripts

Supporting studies that sit outside the main train → evaluate → plot pipeline.
They exist to check this project's own claims, and several of them found
problems that changed the conclusions. Each is standalone and portable: run it
from anywhere, it resolves the repository root from its own path.

| Script | What it answers | Needs |
|---|---|---|
| `fair_baseline_v2.py` | Does the corruption actually hurt, and does the label-free baseline detect it too? | GPU, ~40 min |
| `verify_claims.py` | Do the committed runs support the claims made about them? | `results/`, `data/` |
| `card_numbers.py` | Every headline figure, recomputed from `results/` | `results/` |
| `bench_geometry.py` | What do the geometry metrics cost as embeddings scale? | CPU, ~5 min |
| `time_epoch.py` | Real wall-clock and peak GPU memory per epoch | GPU, ~10 min |
| `rng_workers_demo.py` | Does the shortcut-patch RNG survive DataLoader workers? | CPU, seconds |

---

## `fair_baseline_v2.py` — the comparison `train.py` does not make

`train.py --condition dist_shift` measures geometry on the corrupted stream but
accuracy and confidence on the clean test set. That is not like-for-like, and it
leaves the central question unmeasured: *how much damage does the corruption
actually do?*

This script trains ResNet-18 with the identical protocol and measures every
signal on **both** streams each epoch, plus per-sample OOD AUROC at epochs 1
and 20 for max softmax probability and for distance to the clean-embedding
centroid.

Findings (3 seeds):

- Validation accuracy stays at ~74% while accuracy on the corrupted stream is
  ~39%. The damage is real and large, and the clean-test metric is blind to it.
- Max softmax probability on the corrupted stream **does** separate. Mean
  dispersion shifts ~3× more than mean confidence, but per-sample AUROC is
  0.70 for confidence against 0.76 for centroid distance — a much narrower gap,
  and the direction of the centroid-distance score has to be chosen after
  seeing the data.
- The per-image pixel-std detector in `verify_claims.py` §10 reaches AUROC
  0.986 on the same corruption, with no model at all.

Together these say: the blindness of validation accuracy is the solid result;
"only geometry reacts" is not.

## `verify_claims.py` — the verification suite

Twelve checks against the committed runs and the code. The ones that changed
conclusions:

- **§3** The original detection metric (`evaluate.py:first_change`) fires on all
  40 clean reference runs, usually by the second epoch. It measured warm-up, not
  failure, and the lead-time claim built on it was retracted.
- **§4–5** Out-of-sample false alarms: 1 crossing in 800 checks (200 clean
  epochs × 4 signals), from effective rank; dispersion none. But the onset runs
  reuse the reference runs' seeds, so "held out" is weaker than it sounds —
  same-index pairs are measurably closer than cross-index pairs.
- **§8–9** Two silent failure modes in the analysis code itself: one NaN
  disables detection without error, and `load_runs` stops at the first missing
  run index.
- **§6–7, §10** Three gaps between the code and its description: the label-noise
  rate is ~27% rather than 30%, classes 0 and 9 get the same shortcut colour,
  and `dist_shift` zeroes ~52% of each image rather than adding mild noise.

## `bench_geometry.py` — scaling cost

`effective_rank` calls `np.linalg.svd(E, full_matrices=False)` and discards the
singular vectors. The obvious fix, `compute_uv=False`, is **far slower** for
tall matrices on this LAPACK build (2,645 s versus 77 s at 100k × 512). The
Gram-matrix route — eigenvalues of `EᵀE` — is both fastest and streamable in
O(d²) memory, and is the right change if the test set grows.

On synthetic dense matrices the SVD dominates. On real post-ReLU embeddings it
does not: all geometry is 1.4% of a SimpleCNN epoch and 4.1% of a ResNet-18
epoch (0.24 s and 0.64 s per `effective_rank` call). The bottleneck is
elsewhere — see `time_epoch.py`.

## `time_epoch.py` — where an epoch actually goes

Per-epoch cost is 9.6 s (MNIST-CNN), 44.0 s (CIFAR-CNN) and 48.8 s
(ResNet-18); ResNet-18 has ten times the parameters of SimpleCNN but is only
11% slower per epoch, because host-side work dominates.

For a CIFAR-10 epoch, most of the time is DataLoader worker start-up:
`num_workers=4` without `persistent_workers`, three loader iterations per
epoch, so twelve process starts. Each re-imports torch and is sent the dataset
by pickle — CIFAR-10's images are a NumPy array copied by value (29.3 MiB per
worker for the test split), while MNIST's tensor goes through shared memory.
Measured: ~11.4 s before the first CIFAR batch, against 0.33 s for the GPU
forward pass over the whole test set. `persistent_workers=True` drops later
passes to 0.4 s.

## `rng_workers_demo.py` — a reproducibility bug in the shortcut condition

`SpuriousDataset` draws from a generator stored on the dataset instance, inside
`__getitem__`. With `num_workers > 0` each worker gets a pickled copy of that
generator, so every worker replays the same stream and the stream restarts each
epoch. The patch pattern is therefore fixed across epochs and correlated across
workers, instead of being redrawn per sample.

The single-process path behaves as intended, which is why this is easy to miss.
It does not invalidate the spurious-feature result — the shortcut is still
present in ~90% of training images — but the randomisation is not what the code
claims. Fix: seed from `torch.utils.data.get_worker_info()`, or decide patch
assignment once in `__init__`.
