import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from l4s.bands import L4S_BANDS, prithvi_mapping, band_stats, heuristic_checks
from l4s.features import pixel_features, sample_pixels


def test_prithvi_mapping_substitutes_b8a():
    idx, notes = prithvi_mapping()
    assert len(idx) == 6 and idx[3] == L4S_BANDS.index("B08")
    assert any("B8A" in n for n in notes)


def test_features_shape_and_sampling():
    img = np.random.default_rng(0).random((16, 16, 14)).astype(np.float32)
    m = np.zeros((16, 16), np.uint8); m[:4, :4] = 1
    assert pixel_features(img).shape == (256, 17)
    X, y = sample_pixels(img, m, 40, 0.5)
    assert X.shape == (40, 17) and y.sum() == 16  # all 16 positives (<20 requested)


def test_heuristics_run():
    img = np.random.default_rng(0).random((4, 8, 8, 14))
    assert len(heuristic_checks(band_stats(img))) == 3
