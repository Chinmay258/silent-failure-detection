"""
Cost of geometry.py at current and scaled embedding sizes.

effective_rank() calls np.linalg.svd(E, full_matrices=False), which computes
the left and right singular vectors as well as the singular values. Only the
singular values are used. This benchmarks three ways of getting them:

  full SVD     what geometry.py does today
  svals only   np.linalg.svd(..., compute_uv=False) - the obvious fix, which
               on this LAPACK build is dramatically SLOWER for tall matrices
  Gram matrix  eigenvalues of E^T E - O(d^2) memory, accumulable batch by batch

Usage:
    python analysis/bench_geometry.py
"""

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np

from geometry import dispersion, effective_rank, min_class_effective_rank


def erank_svals_only(E):
    s = np.linalg.svd(E, compute_uv=False)
    s = s[s > 1e-10]
    p = s / s.sum()
    return float(np.exp(-np.sum(p * np.log(p + 1e-10))))


def erank_gram(E):
    """Eigenvalues of the d x d Gram matrix; O(d^2) memory, streamable."""
    ev = np.linalg.eigvalsh(E.T.astype(np.float64) @ E.astype(np.float64))
    s = np.sqrt(np.clip(ev, 0, None))
    s = s[s > 1e-10]
    p = s / s.sum()
    return float(np.exp(-np.sum(p * np.log(p + 1e-10))))


def timed(fn, *a, reps=3):
    best, v = float("inf"), None
    for _ in range(reps):
        t = time.perf_counter()
        v = fn(*a)
        best = min(best, time.perf_counter() - t)
    return best, v


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print(f"{'N':>7} {'d':>5} {'MB':>6} | {'dispersion':>10} {'erank(full SVD)':>16} "
          f"{'erank(svals)':>13} {'erank(gram)':>12} | agree")
    for N, d, reps in [(10_000, 256, 3), (10_000, 512, 3),
                       (100_000, 512, 1), (10_000, 2048, 1)]:
        # non-negative, like post-ReLU features
        E = np.abs(rng.standard_normal((N, d), dtype=np.float32))
        td, _ = timed(dispersion, E, reps=reps)
        tf, vf = timed(effective_rank, E, reps=reps)
        ts, vs = timed(erank_svals_only, E, reps=reps)
        tg, vg = timed(erank_gram, E, reps=reps)
        print(f"{N:>7} {d:>5} {E.nbytes / 2**20:>6.0f} | {td:>9.3f}s {tf:>15.3f}s "
              f"{ts:>12.3f}s {tg:>11.3f}s | "
              f"{abs(vf - vs) < 1e-3 and abs(vf - vg) < 1e-2}")

    y = rng.integers(0, 10, size=10_000)
    E = np.abs(rng.standard_normal((10_000, 512), dtype=np.float32))
    t, _ = timed(min_class_effective_rank, E, y)
    print(f"min_class_effective_rank N=10000 d=512 (10 SVDs): {t:.3f}s")
    print("\nNote: these are dense random matrices. Real post-ReLU embeddings are "
          "sparser and lower-rank;\nsee analysis/README.md for the measured share "
          "of a real training epoch.")
