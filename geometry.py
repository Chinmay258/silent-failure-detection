import numpy as np


def dispersion(E: np.ndarray) -> float:
    """Mean distance of embeddings from their centroid."""
    mu = E.mean(axis=0)
    return float(np.mean(np.linalg.norm(E - mu, axis=1)))


def effective_rank(E: np.ndarray) -> float:
    """Exponential of the Shannon entropy of the singular value distribution."""
    _, s, _ = np.linalg.svd(E, full_matrices=False)
    s = s[s > 1e-10]
    p = s / s.sum()
    return float(np.exp(-np.sum(p * np.log(p + 1e-10))))


def within_class_dispersion(E: np.ndarray, y: np.ndarray) -> float:
    """Mean over classes of the average distance to the class centroid.

    Class-conditional counterpart of dispersion(): sensitive to individual
    classes collapsing even when the global embedding spread looks normal.
    """
    vals = []
    for c in np.unique(y):
        Ec = E[y == c]
        if len(Ec) < 2:
            continue
        mu = Ec.mean(axis=0)
        vals.append(np.mean(np.linalg.norm(Ec - mu, axis=1)))
    return float(np.mean(vals))


def min_class_effective_rank(E: np.ndarray, y: np.ndarray) -> float:
    """Effective rank of the most collapsed class's embeddings.

    Targets failures that hit a subset of classes (e.g. minority classes
    under imbalance) and are invisible to the global effective rank.
    """
    vals = []
    for c in np.unique(y):
        Ec = E[y == c]
        if len(Ec) < 2:
            continue
        vals.append(effective_rank(Ec))
    return float(min(vals))
