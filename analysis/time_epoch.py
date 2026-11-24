"""
Wall-clock per epoch and peak GPU memory for each architecture/dataset pairing,
using the real train.train_one_run (DataLoader num_workers=4, pinned memory,
per-epoch eval plus embedding extraction plus SVD).

Marginal epoch time = t(2 epochs) - t(1 epoch), which cancels the fixed
start-up cost (dataset construction and its MD5 check, worker spawn, CUDA
context creation, model init).

Writes its throwaway metrics to a temp directory, not to results/.

Usage:
    python analysis/time_epoch.py
"""

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)

import torch

import train

if __name__ == "__main__":
    out = tempfile.mkdtemp(prefix="sfd_timing_")
    try:
        for cond in ["clean_mnist", "clean_cifar", "clean_resnet"]:
            t, peak = {}, {}
            for ep in (1, 2):
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                train.train_one_run(cond, ep, 0, out)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t[ep] = time.perf_counter() - t0
                peak[ep] = (torch.cuda.max_memory_allocated() / 2**20
                            if torch.cuda.is_available() else 0.0)
            marginal = t[2] - t[1]
            print(f"TIMING {cond}: run(1 ep)={t[1]:.1f}s run(2 ep)={t[2]:.1f}s "
                  f"-> per-epoch {marginal:.1f}s, fixed overhead {t[1] - marginal:.1f}s, "
                  f"peak GPU alloc {max(peak.values()):.0f} MiB", flush=True)
    finally:
        shutil.rmtree(out, ignore_errors=True)
    print("TIMING DONE", flush=True)
