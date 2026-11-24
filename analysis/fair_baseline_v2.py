"""
Same-stream comparison under input corruption.

The main experiment (train.py --condition dist_shift) measures geometry on the
corrupted stream but accuracy and confidence on the clean test set, which is
not a like-for-like comparison. Two things are therefore untested there:

  1. Ground-truth harm. Clean-test accuracy barely moves under dist_shift, so
     the size of the damage is never measured. Accuracy ON the corrupted
     stream is the missing number.
  2. The label-free baseline. Max softmax probability computed on the SAME
     corrupted stream needs no labels and is the standard OOD baseline
     (Hendrycks & Gimpel, ICLR 2017). If it separates as well as dispersion,
     "only geometry reacts" is false.

Protocol is identical to dist_shift / clean_resnet: ResNet-18 from scratch,
Adam 1e-3, batch 256, 20 epochs, normalized CIFAR-10, seeds 0/42/84. Data is
held on the GPU to avoid DataLoader worker start-up (see analysis/README.md).

Every epoch, every signal is measured on BOTH streams. At epochs 1 and 20 a
per-sample OOD AUROC is also computed for max softmax probability and for
distance to the clean-embedding centroid (the per-sample analogue of
dispersion), because a ratio of batch averages is not how an OOD baseline is
normally scored.

Writes results/fair_baseline_v2/seed<N>.npy.

Usage:
    python analysis/fair_baseline_v2.py
"""

import os
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.chdir(REPO)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from geometry import dispersion, effective_rank
from train import ResNet18Wrapper, add_gaussian_noise

OUT = REPO / "results" / "fair_baseline_v2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


def load_split(train: bool):
    """CIFAR-10 as one normalized GPU tensor, matching train.py's transform."""
    ds = torchvision.datasets.CIFAR10("data", train=train, download=True)
    x = torch.from_numpy(ds.data).float().permute(0, 3, 1, 2) / 255.0
    return ((x - MEAN) / STD).to(DEVICE), torch.tensor(ds.targets).to(DEVICE)


def auroc(score_id, score_ood):
    """P(score_ood > score_id), rank-based (Mann-Whitney U)."""
    s = np.r_[score_id, score_ood]
    r = np.empty(len(s))
    r[s.argsort()] = np.arange(1, len(s) + 1)
    n1, n2 = len(score_id), len(score_ood)
    return float((r[n1:].sum() - n2 * (n2 + 1) / 2) / (n1 * n2))


@torch.no_grad()
def forward_all(model, Xte, Yte, noised: bool):
    model.eval()
    correct, msp, embs = 0, [], []
    for i in range(0, Xte.shape[0], 512):
        x, y = Xte[i:i + 512], Yte[i:i + 512]
        if noised:
            x = add_gaussian_noise(x, std=0.1)
        logits, emb = model(x, return_embedding=True)
        p = torch.softmax(logits, dim=1)
        correct += (p.argmax(dim=1) == y).sum().item()
        msp.append(p.max(dim=1).values.cpu().numpy())
        embs.append(emb.cpu().numpy())
    return correct / Xte.shape[0], np.concatenate(msp), np.concatenate(embs)


def run(seed: int, Xtr, Ytr, Xte, Yte, epochs: int = 20, batch: int = 256):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = ResNet18Wrapper().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    m = {f"{s}_{t}": [] for t in ("clean", "noised")
         for s in ("acc", "conf", "disp", "erank")}
    m["auroc_msp"], m["auroc_centroid_dist"] = {}, {}
    n = Xtr.shape[0]
    t0 = time.perf_counter()

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            crit(model(Xtr[idx]), Ytr[idx]).backward()
            opt.step()

        res = {}
        for tag, noised in (("clean", False), ("noised", True)):
            acc, msp, E = forward_all(model, Xte, Yte, noised)
            res[tag] = (msp, E)
            m[f"acc_{tag}"].append(acc)
            m[f"conf_{tag}"].append(float(msp.mean()))
            m[f"disp_{tag}"].append(dispersion(E))
            m[f"erank_{tag}"].append(effective_rank(E))

        if ep in (0, epochs - 1):
            (msp_c, E_c), (msp_n, E_n) = res["clean"], res["noised"]
            mu = E_c.mean(axis=0)
            m["auroc_msp"][ep + 1] = auroc(-msp_c, -msp_n)
            m["auroc_centroid_dist"][ep + 1] = auroc(
                np.linalg.norm(E_c - mu, axis=1), np.linalg.norm(E_n - mu, axis=1))

        print(f"seed {seed} ep {ep + 1:>2} | clean acc={m['acc_clean'][-1]:.4f} "
              f"disp={m['disp_clean'][-1]:.2f} | corrupted acc={m['acc_noised'][-1]:.4f} "
              f"conf={m['conf_noised'][-1]:.4f} disp={m['disp_noised'][-1]:.2f} | "
              f"{time.perf_counter() - t0:.0f}s", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / f"seed{seed}.npy", m)
    return m


def summarize(ms):
    g = lambda k: np.array([mm[k] for mm in ms])
    ac, an = g("acc_clean"), g("acc_noised")
    print("\n=== SUMMARY (3 seeds) ===")
    print(f"acc clean     ep1 {ac[:, 0].mean():.3f}  ep20 {ac[:, -1].mean():.3f} +/- {ac[:, -1].std():.3f}")
    print(f"acc corrupted ep1 {an[:, 0].mean():.3f}  ep20 {an[:, -1].mean():.3f} +/- {an[:, -1].std():.3f}")
    gap = 100 * (ac[:, -1] - an[:, -1])
    print(f"gap at ep20: {gap.mean():.1f} +/- {gap.std():.1f} points")
    rel = {}
    for name in ("conf", "disp", "erank"):
        cl, no = g(f"{name}_clean"), g(f"{name}_noised")
        per_epoch = np.abs(no - cl) / np.abs(cl)
        rel[name] = per_epoch.mean(axis=1)
        print(f"{name:<6} relative separation {100 * rel[name].mean():5.1f}% "
              f"+/- {100 * rel[name].std():.1f}  (ep1 {100 * per_epoch[:, 0].mean():.1f}%, "
              f"ep20 {100 * per_epoch[:, -1].mean():.1f}%)")
    ratio = rel["disp"] / rel["conf"]
    print(f"dispersion/confidence relative-separation ratio: {ratio.mean():.2f} +/- {ratio.std():.2f}")
    for key in ("auroc_msp", "auroc_centroid_dist"):
        for ep in (1, 20):
            v = np.array([mm[key][ep] for mm in ms])
            print(f"{key:<20} epoch {ep:>2}: AUROC {v.mean():.4f} +/- {v.std():.4f}  "
                  f"(two-sided max(a,1-a) = {np.maximum(v, 1 - v).mean():.4f})")


if __name__ == "__main__":
    Xtr, Ytr = load_split(True)
    Xte, Yte = load_split(False)
    summarize([run(s, Xtr, Ytr, Xte, Yte) for s in (0, 42, 84)])
    print("FAIR BASELINE V2 DONE", flush=True)
