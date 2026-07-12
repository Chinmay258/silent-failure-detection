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

from geometry import dispersion, effective_rank

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
    """Return embedding matrix (N × D) for the full test set."""
    model.eval()
    embs = []
    for x, _ in loader:
        x = x.to(device)
        if condition == "dist_shift":
            # apply OOD noise at eval time for dist_shift condition
            x = add_gaussian_noise(x, std=0.1)
        _, emb = model(x, return_embedding=True)
        embs.append(emb.cpu().numpy())
    return np.concatenate(embs, axis=0)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_run(condition: str, epochs: int, run: int, out_dir: str):
    seed = run * 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, model = build_condition(condition, run)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    metrics = {
        "accuracy":       [],
        "confidence":     [],
        "dispersion":     [],
        "effective_rank": [],
    }

    for epoch in range(epochs):
        # --- train ---
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        # --- eval ---
        model.eval()
        correct, total, conf_sum = 0, 0, 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                probs  = torch.softmax(logits, dim=1)
                preds  = probs.argmax(dim=1)
                correct   += (preds == y).sum().item()
                total     += y.size(0)
                conf_sum  += probs.max(dim=1).values.sum().item()

        acc  = correct / total
        conf = conf_sum / total

        # --- geometry ---
        E    = extract_embeddings(model, test_loader, device, condition)
        disp = dispersion(E)
        rank = effective_rank(E)

        metrics["accuracy"].append(acc)
        metrics["confidence"].append(conf)
        metrics["dispersion"].append(disp)
        metrics["effective_rank"].append(rank)

        print(f"  Run {run} | Epoch {epoch+1:>2}/{epochs} | "
              f"Acc={acc:.3f}  Conf={conf:.3f}  "
              f"Disp={disp:.2f}  ERank={rank:.2f}")

    # save
    run_dir = os.path.join(out_dir, condition, f"run{run}")
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
    args = parser.parse_args()

    print(f"\n=== Condition: {args.condition} | {args.runs} runs × {args.epochs} epochs ===\n")
    for run in range(args.runs):
        train_one_run(args.condition, args.epochs, run, args.out)


if __name__ == "__main__":
    main()
