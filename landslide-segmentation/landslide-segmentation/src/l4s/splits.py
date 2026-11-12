"""Leakage-aware splits and pseudo-regions (numpy-only k-means)."""
import numpy as np


def patch_descriptors(X):
    """X: (N, H, W, C) -> (N, 2C) per-patch mean and std (standardised)."""
    d = np.concatenate([X.mean((1, 2)), X.std((1, 2))], 1)
    return (d - d.mean(0)) / (d.std(0) + 1e-6)


def kmeans(D, k, iters=100, seed=0):
    rng = np.random.default_rng(seed)
    # k-means++ init
    C = [D[rng.integers(len(D))]]
    for _ in range(1, k):
        dist = np.min(((D[:, None] - np.array(C)[None]) ** 2).sum(-1), 1)
        C.append(D[rng.choice(len(D), p=dist / dist.sum())])
    C = np.array(C)
    for _ in range(iters):
        lab = ((D[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        newC = np.array([D[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    return lab


def cluster_split(labels, val_clusters, seed=0):
    """Hold out whole clusters as validation. Returns (train_idx, val_idx, held_out_ids)."""
    rng = np.random.default_rng(seed)
    ids = np.unique(labels)
    held = rng.choice(ids, size=val_clusters, replace=False)
    val = np.flatnonzero(np.isin(labels, held))
    tr = np.flatnonzero(~np.isin(labels, held))
    return tr, val, sorted(int(h) for h in held)


def leave_one_region_out(labels):
    for r in np.unique(labels):
        yield int(r), np.flatnonzero(labels != r), np.flatnonzero(labels == r)
