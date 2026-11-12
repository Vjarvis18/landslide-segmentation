"""Per-pixel features for the RF/LightGBM baseline."""
import numpy as np
from .bands import L4S_BANDS, band_index


def _nd(a, b, eps=1e-6):
    return (a - b) / (a + b + eps)


def pixel_features(img, bands=L4S_BANDS):
    """img: (H, W, C) -> (H*W, C+3) with NDVI, NDWI, brightness appended."""
    nir = img[..., band_index("B08", bands)]
    red = img[..., band_index("B04", bands)]
    grn = img[..., band_index("B03", bands)]
    ndvi = _nd(nir, red)
    ndwi = _nd(grn, nir)
    bright = img[..., :12].mean(-1)
    feats = np.concatenate([img, ndvi[..., None], ndwi[..., None], bright[..., None]], -1)
    return feats.reshape(-1, feats.shape[-1])


def sample_pixels(img, mask, n, landslide_fraction=0.5, rng=None):
    """Stratified pixel sampling to counter extreme class imbalance."""
    rng = rng or np.random.default_rng(0)
    X = pixel_features(img)
    y = mask.reshape(-1)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    n_pos = min(len(pos), int(n * landslide_fraction))
    n_neg = min(len(neg), n - n_pos)
    sel = np.concatenate([
        rng.choice(pos, n_pos, replace=False) if n_pos else np.empty(0, int),
        rng.choice(neg, n_neg, replace=False) if n_neg else np.empty(0, int),
    ])
    return X[sel], y[sel]
