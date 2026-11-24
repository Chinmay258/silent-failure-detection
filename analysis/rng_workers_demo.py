"""
Does SpuriousDataset's per-instance RNG survive DataLoader multiprocessing?

SpuriousDataset draws self.rng.random() inside __getitem__ to decide whether a
sample gets the shortcut patch. With num_workers > 0 each worker process gets a
pickled COPY of the dataset, including the generator state, so:

  - every worker replays the SAME random stream, and
  - the stream restarts every epoch, because the workers are respawned.

The patch pattern is therefore identical across epochs and correlated across
workers, instead of being redrawn per sample as intended. The single-process
path (num_workers=0) behaves as intended, which is why this is easy to miss.

Run on a tiny synthetic dataset so it finishes instantly.

Usage:
    python analysis/rng_workers_demo.py
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch
from torch.utils.data import DataLoader, Dataset

from train import SpuriousDataset


class Blank(Dataset):
    """64 all-zero 3x8x8 images, so any non-zero corner must be the patch."""

    def __len__(self):
        return 64

    def __getitem__(self, i):
        return torch.zeros(3, 8, 8), i % 10


def patch_mask(loader):
    """Per-sample patch presence, in loader order (shuffle=False => index order)."""
    out = []
    for imgs, _ in loader:
        out.extend((imgs[:, :, :4, :4].abs().sum(dim=(1, 2, 3)) > 0).int().tolist())
    return out


if __name__ == "__main__":
    print("multiprocessing start method:", torch.multiprocessing.get_start_method())

    ds = SpuriousDataset(Blank(), correlation=0.5, seed=0)
    single = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    s1, s2 = patch_mask(single), patch_mask(single)
    print(f"num_workers=0: epoch1 == epoch2 ? {s1 == s2}   "
          f"(patched {sum(s1)}/64, {sum(s2)}/64)   <- intended: RNG advances")

    ds2 = SpuriousDataset(Blank(), correlation=0.5, seed=0)
    multi = DataLoader(ds2, batch_size=8, shuffle=False, num_workers=2)
    m1, m2 = patch_mask(multi), patch_mask(multi)
    print(f"num_workers=2: epoch1 == epoch2 ? {m1 == m2}   "
          f"(patched {sum(m1)}/64, {sum(m2)}/64)   <- bug: identical every epoch")
    b = [m1[i:i + 8] for i in range(0, 64, 8)]
    print("num_workers=2: batches from different workers identical ?",
          [b[k] == b[k + 1] for k in range(0, 8, 2)])
    print("\nFix: draw from torch.utils.data.get_worker_info()-seeded randomness, "
          "or decide the\npatch once at __init__ instead of inside __getitem__.")
