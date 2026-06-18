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
