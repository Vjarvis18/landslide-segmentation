import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from l4s.splits import patch_descriptors, kmeans, cluster_split, leave_one_region_out


def test_cluster_split_disjoint():
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(0, 1, (20, 8, 8, 14)), rng.normal(5, 1, (20, 8, 8, 14))])
    lab = kmeans(patch_descriptors(X), 4, seed=1)
    tr, va, held = cluster_split(lab, 1, seed=0)
    assert len(set(tr) & set(va)) == 0
    assert len(tr) + len(va) == len(X)
    assert set(np.unique(lab[va])) == set(held)


def test_kmeans_separates_blobs():
    rng = np.random.default_rng(0)
    D = np.concatenate([rng.normal(0, .1, (30, 2)), rng.normal(10, .1, (30, 2))])
    lab = kmeans(D, 2)
    assert len(set(lab[:30])) == 1 and len(set(lab[30:])) == 1 and lab[0] != lab[-1]


def test_loro():
    lab = np.array([0, 0, 1, 2, 2])
    folds = list(leave_one_region_out(lab))
    assert len(folds) == 3
