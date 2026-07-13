"""
train.py
--------
Unified training engine for all four failure conditions.

Usage:
    python train.py --condition label_noise    --epochs 20 --runs 5
    python train.py --condition spurious       --epochs 20 --runs 5
    python train.py --condition dist_shift     --epochs 20 --runs 5
    python train.py --condition class_imbalance --epochs 20 --runs 5
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

from geometry import (dispersion, effective_rank,
                      within_class_dispersion, min_class_effective_rank)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class SimpleCNN(nn.Module):
    """Lightweight CNN for MNIST / CIFAR-10."""
    def __init__(self, in_channels: int, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # figure out flattened size
        dummy_h = 28 if in_channels == 1 else 32
        dummy = torch.zeros(1, in_channels, dummy_h, dummy_h)
        flat = self.features(dummy).view(1, -1).shape[1]

        self.penultimate = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 256), nn.ReLU(),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x, return_embedding: bool = False):
        x = self.features(x)
        emb = self.penultimate(x)
        logits = self.classifier(emb)
        if return_embedding:
            return logits, emb
        return logits


class ResNet18Wrapper(nn.Module):
    """ResNet-18 with accessible penultimate layer."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        base = torchvision.models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # drop fc
        self.penultimate_act = nn.Identity()
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x, return_embedding: bool = False):
        emb = self.backbone(x).flatten(1)
        emb = self.penultimate_act(emb)
        logits = self.classifier(emb)
        if return_embedding:
            return logits, emb
        return logits


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

class NoisyLabelDataset(Dataset):
    """Wraps a dataset and randomly flips `noise_rate` fraction of labels."""
    def __init__(self, base_dataset, noise_rate: float = 0.3, seed: int = 0):
        self.dataset = base_dataset
        rng = np.random.default_rng(seed)
        n = len(base_dataset)
        self.labels = np.array([base_dataset[i][1] for i in range(n)])
        num_classes = int(self.labels.max()) + 1
        flip_idx = rng.choice(n, size=int(noise_rate * n), replace=False)
        self.labels[flip_idx] = rng.integers(0, num_classes, size=len(flip_idx))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        return img, int(self.labels[idx])


class SpuriousDataset(Dataset):
    """
    Adds a small coloured patch whose hue is correlated with the class label.
    The patch is present in `correlation` fraction of training samples.
    """
    def __init__(self, base_dataset, correlation: float = 0.9, patch_size: int = 4, seed: int = 0):
        self.dataset = base_dataset
        self.correlation = correlation
        self.patch_size = patch_size
        self.rng = np.random.default_rng(seed)
        # fixed hue per class (10 classes → 10 colours)
        self.class_colors = np.linspace(0, 1, 10)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]           # img: C×H×W tensor in [0,1]
        img = img.clone()
        if self.rng.random() < self.correlation:
            p = self.patch_size
            # map hue → RGB via a simple formula
            h = self.class_colors[label]
            r = abs(h * 6 - 3) - 1
            g = 2 - abs(h * 6 - 2)
            b = 2 - abs(h * 6 - 4)
            color = torch.tensor([r, g, b], dtype=torch.float32).clamp(0, 1)
            for c in range(min(img.shape[0], 3)):
                img[c, :p, :p] = color[c]
        return img, label


def add_gaussian_noise(x: torch.Tensor, std: float = 0.1) -> torch.Tensor:
    return (x + torch.randn_like(x) * std).clamp(0, 1)


class ImbalancedDataset(Dataset):
    """
    Severe class imbalance: majority_classes get `majority_frac` of all
    training samples; the remaining classes share the rest equally.

    Default: classes 0-1 get 90% of data, classes 2-9 share 10%.
    The model achieves high accuracy on majority classes while silently
    failing on minority ones — overall accuracy stays deceptively high.
    """
    def __init__(self, base_dataset, majority_classes=(0, 1),
                 majority_frac: float = 0.9, seed: int = 0):
        rng = np.random.default_rng(seed)
        labels = np.array([base_dataset[i][1] for i in range(len(base_dataset))])

        # split indices by majority vs minority
        maj_idx = np.where(np.isin(labels, majority_classes))[0]
        min_idx = np.where(~np.isin(labels, majority_classes))[0]

        n_total   = len(base_dataset)
        n_majority = int(n_total * majority_frac)
        n_minority = n_total - n_majority

        # sample with replacement to hit exact counts
        sampled_maj = rng.choice(maj_idx, size=n_majority, replace=True)
        sampled_min = rng.choice(min_idx, size=n_minority, replace=True)

        self.indices = np.concatenate([sampled_maj, sampled_min])
        rng.shuffle(self.indices)
        self.dataset = base_dataset

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[int(self.indices[idx])]


# ---------------------------------------------------------------------------
# Build datasets & model
# ---------------------------------------------------------------------------

def build_condition(condition: str, run: int):
    """Return (train_loader, test_loader, model, device_str)."""

    if condition == "label_noise":
        tf = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
        base_train = torchvision.datasets.MNIST("data", train=True,  download=True, transform=tf)
        base_test  = torchvision.datasets.MNIST("data", train=False, download=True, transform=tf)
        train_ds   = NoisyLabelDataset(base_train, noise_rate=0.3, seed=run)
        test_ds    = base_test
        model      = SimpleCNN(in_channels=1)

    elif condition == "spurious":
        tf = T.Compose([T.ToTensor()])
        base_train = torchvision.datasets.CIFAR10("data", train=True,  download=True, transform=tf)
        base_test  = torchvision.datasets.CIFAR10("data", train=False, download=True, transform=tf)
        train_ds   = SpuriousDataset(base_train, correlation=0.9, seed=run)
        test_ds    = base_test                      # test set has NO spurious patch
        model      = SimpleCNN(in_channels=3)

    elif condition == "dist_shift":
        tf = T.Compose([T.ToTensor(),
                         T.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2470, 0.2435, 0.2616))])
        base_train = torchvision.datasets.CIFAR10("data", train=True,  download=True, transform=tf)
        base_test  = torchvision.datasets.CIFAR10("data", train=False, download=True, transform=tf)
        train_ds   = base_train
        test_ds    = base_test
        model      = ResNet18Wrapper()

    elif condition == "class_imbalance":
        tf = T.Compose([T.ToTensor(),
                        T.Normalize((0.4914, 0.4822, 0.4465),
                                    (0.2470, 0.2435, 0.2616))])
        base_train = torchvision.datasets.CIFAR10("data", train=True,  download=True, transform=tf)
        base_test  = torchvision.datasets.CIFAR10("data", train=False, download=True, transform=tf)
        # classes 0 (airplane) & 1 (automobile) dominate: 90% of training data
        train_ds   = ImbalancedDataset(base_train, majority_classes=(0, 1),
                                       majority_frac=0.9, seed=run)
        test_ds    = base_test          # balanced test set → accuracy looks fine overall
        model      = SimpleCNN(in_channels=3)

    # ---- clean-training controls -------------------------------------------
    # One uncorrupted reference per failure condition, matching its exact data
    # pipeline, so failure-run trajectories can be compared against a
    # clean-reference band (see evaluate_ref.py):
    #   clean_mnist     -> reference for label_noise      (normalized MNIST, CNN)
    #   clean_cifar_raw -> reference for spurious         (unnormalized CIFAR, CNN)
    #   clean_cifar     -> reference for class_imbalance  (normalized CIFAR, CNN)
    #   clean_resnet    -> reference for dist_shift       (normalized CIFAR, ResNet-18)

    elif condition == "clean_mnist":
        tf = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
        train_ds = torchvision.datasets.MNIST("data", train=True,  download=True, transform=tf)
        test_ds  = torchvision.datasets.MNIST("data", train=False, download=True, transform=tf)
        model    = SimpleCNN(in_channels=1)

    elif condition == "clean_cifar_raw":
        tf = T.Compose([T.ToTensor()])
        train_ds = torchvision.datasets.CIFAR10("data", train=True,  download=True, transform=tf)
        test_ds  = torchvision.datasets.CIFAR10("data", train=False, download=True, transform=tf)
        model    = SimpleCNN(in_channels=3)

    elif condition == "clean_cifar":
        tf = T.Compose([T.ToTensor(),
                        T.Normalize((0.4914, 0.4822, 0.4465),
                                    (0.2470, 0.2435, 0.2616))])
        train_ds = torchvision.datasets.CIFAR10("data", train=True,  download=True, transform=tf)
        test_ds  = torchvision.datasets.CIFAR10("data", train=False, download=True, transform=tf)
        model    = SimpleCNN(in_channels=3)

    elif condition == "clean_resnet":
        tf = T.Compose([T.ToTensor(),
                        T.Normalize((0.4914, 0.4822, 0.4465),
                                    (0.2470, 0.2435, 0.2616))])
        train_ds = torchvision.datasets.CIFAR10("data", train=True,  download=True, transform=tf)
        test_ds  = torchvision.datasets.CIFAR10("data", train=False, download=True, transform=tf)
        model    = ResNet18Wrapper()

    else:
        raise ValueError(f"Unknown condition: {condition}")

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False,
                              num_workers=4, pin_memory=True)
    return train_loader, test_loader, model


# ---------------------------------------------------------------------------
# Geometry extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(model, loader, device, condition):
    """Return (embedding matrix N × D, labels N) for the full test set."""
    model.eval()
    embs, ys = [], []
    for x, y in loader:
        x = x.to(device)
        if condition == "dist_shift":
            # apply OOD noise at eval time for dist_shift condition
            x = add_gaussian_noise(x, std=0.1)
        _, emb = model(x, return_embedding=True)
        embs.append(emb.cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(embs, axis=0), np.concatenate(ys, axis=0)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

# clean counterpart of each failure condition (same data pipeline & model);
# used by --onset to train clean before the corruption switches on.
# Keep in sync with PAIRS in evaluate_ref.py.
CLEAN_REFERENCE = {
    "label_noise":     "clean_mnist",
    "spurious":        "clean_cifar_raw",
    "class_imbalance": "clean_cifar",
    "dist_shift":      "clean_resnet",
}


def train_one_run(condition: str, epochs: int, run: int, out_dir: str,
                  onset: int = None):
    """
    Train one seed and record per-epoch monitoring signals.

    onset: if set, the first `onset` epochs train on CLEAN data and the
    corruption switches on afterwards (for dist_shift, the evaluation-time
    input noise starts after `onset` instead). Simulates a failure that
    begins mid-training, so detection delay is measurable for every
    condition. Results are saved under "<condition>_onset<k>".
    """
    seed = run * 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, model = build_condition(condition, run)
    if onset is not None:
        clean_train_loader, _, _ = build_condition(CLEAN_REFERENCE[condition], run)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    metrics = {
        "accuracy":       [],
        "confidence":     [],
        "dispersion":     [],
        "effective_rank": [],
        # baselines & class-conditional signals (v2)
        "train_loss":     [],
        "test_loss":      [],
        "grad_norm":      [],
        "min_class_acc":  [],
        "within_class_dispersion": [],
        "min_class_erank": [],
    }

    for epoch in range(epochs):
        corrupted = onset is None or epoch >= onset
        loader = train_loader if corrupted else clean_train_loader
        extract_cond = condition if corrupted else "clean"

        # --- train ---
        model.train()
        loss_sum = torch.zeros((), device=device)
        gn_sum = torch.zeros((), device=device)
        n_seen, n_steps = 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            gn_sum += torch.sqrt(sum(p.grad.detach().pow(2).sum()
                                     for p in model.parameters()
                                     if p.grad is not None))
            optimizer.step()
            loss_sum += loss.detach() * y.size(0)
            n_seen += y.size(0)
            n_steps += 1
        train_loss = (loss_sum / n_seen).item()
        grad_norm = (gn_sum / n_steps).item()

        # --- eval ---
        model.eval()
        correct, total, conf_sum, tloss_sum = 0, 0, 0.0, 0.0
        cls_correct = np.zeros(10)
        cls_total = np.zeros(10)
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                probs  = torch.softmax(logits, dim=1)
                preds  = probs.argmax(dim=1)
                correct   += (preds == y).sum().item()
                total     += y.size(0)
                conf_sum  += probs.max(dim=1).values.sum().item()
                tloss_sum += nn.functional.cross_entropy(
                    logits, y, reduction="sum").item()
                for c in range(10):
                    m = y == c
                    cls_total[c]   += m.sum().item()
                    cls_correct[c] += (preds[m] == c).sum().item()

        acc  = correct / total
        conf = conf_sum / total
        test_loss = tloss_sum / total
        min_class_acc = float((cls_correct[cls_total > 0]
                               / cls_total[cls_total > 0]).min())

        # --- geometry ---
        E, Ey = extract_embeddings(model, test_loader, device, extract_cond)
        disp = dispersion(E)
        rank = effective_rank(E)

        metrics["accuracy"].append(acc)
        metrics["confidence"].append(conf)
        metrics["dispersion"].append(disp)
        metrics["effective_rank"].append(rank)
        metrics["train_loss"].append(train_loss)
        metrics["test_loss"].append(test_loss)
        metrics["grad_norm"].append(grad_norm)
        metrics["min_class_acc"].append(min_class_acc)
        metrics["within_class_dispersion"].append(within_class_dispersion(E, Ey))
        metrics["min_class_erank"].append(min_class_effective_rank(E, Ey))

        tag = "" if onset is None else ("  [corrupt]" if corrupted else "  [clean]")
        print(f"  Run {run} | Epoch {epoch+1:>2}/{epochs} | "
              f"Acc={acc:.3f}  Conf={conf:.3f}  "
              f"Disp={disp:.2f}  ERank={rank:.2f}{tag}")

    # save
    save_name = condition if onset is None else f"{condition}_onset{onset}"
    run_dir = os.path.join(out_dir, save_name, f"run{run}")
    os.makedirs(run_dir, exist_ok=True)
    np.save(os.path.join(run_dir, "metrics.npy"), metrics)
    print(f"  → Saved to {run_dir}/metrics.npy")
    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True,
                        choices=["label_noise", "spurious", "dist_shift", "class_imbalance",
                                 "clean_mnist", "clean_cifar_raw", "clean_cifar", "clean_resnet"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--runs",   type=int, default=5)
    parser.add_argument("--out",    type=str, default="results")
    parser.add_argument("--onset",  type=int, default=None,
                        help="number of clean epochs before the corruption "
                             "switches on (failure conditions only)")
    args = parser.parse_args()

    if args.onset is not None and args.condition not in CLEAN_REFERENCE:
        raise SystemExit(f"--onset requires a failure condition, "
                         f"not '{args.condition}'")

    onset_txt = "" if args.onset is None else f" | onset at epoch {args.onset}"
    print(f"\n=== Condition: {args.condition} | {args.runs} runs × "
          f"{args.epochs} epochs{onset_txt} ===\n")
    for run in range(args.runs):
        train_one_run(args.condition, args.epochs, run, args.out, onset=args.onset)


if __name__ == "__main__":
    main()
