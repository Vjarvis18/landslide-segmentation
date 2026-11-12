import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from l4s.metrics import scores


def test_perfect():
    m = np.zeros((4, 4)); m[1:3, 1:3] = 1
    s = scores(m, m)
    assert abs(s["f1"] - 1) < 1e-6 and abs(s["iou"] - 1) < 1e-6


def test_known_counts():
    p = np.array([1, 1, 0, 0]); t = np.array([1, 0, 1, 0])
    s = scores(p, t)
    assert (s["tp"], s["fp"], s["fn"], s["tn"]) == (1, 1, 1, 1)
    assert abs(s["iou"] - 1 / 3) < 1e-6 and abs(s["f1"] - 0.5) < 1e-6
