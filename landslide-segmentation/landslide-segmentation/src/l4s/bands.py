"""Landslide4Sense band definitions and band-order sanity checks.

L4S patches are 128 x 128 x 14: 12 Sentinel-2 channels + slope + DEM (ALOS PALSAR).
The commonly cited order is B1..B12 (no B8A); verify against the dataset README.
"""
import numpy as np

# Assumed order. Edit after running scripts/check_bands.py if your inspection disagrees.
L4S_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08",
             "B09", "B10", "B11", "B12", "SLOPE", "DEM"]

PRITHVI_BANDS = ["B02", "B03", "B04", "B8A", "B11", "B12"]


def band_index(name, bands=L4S_BANDS):
    return bands.index(name)


def prithvi_mapping(bands=L4S_BANDS):
    """Return (indices, notes) mapping L4S channels to Prithvi's 6 HLS bands.

    If B8A is absent, B08 is substituted (broad NIR vs narrow NIR) and a note is returned.
    """
    idx, notes = [], []
    for b in PRITHVI_BANDS:
        if b in bands:
            idx.append(bands.index(b))
        elif b == "B8A" and "B08" in bands:
            idx.append(bands.index("B08"))
            notes.append("B8A missing: substituted B08 (broad NIR) for NIR_NARROW")
        else:
            raise ValueError(f"Band {b} unavailable and has no substitute")
    return idx, notes


def band_stats(x):
    """Per-channel stats for an array (N, H, W, C) or (H, W, C)."""
    x = np.asarray(x, dtype=np.float64)
    flat = x.reshape(-1, x.shape[-1])
    return {
        "mean": flat.mean(0), "std": flat.std(0),
        "min": flat.min(0), "max": flat.max(0),
    }


def heuristic_checks(stats, bands=L4S_BANDS):
    """Physics-based sanity checks on mean reflectance of a vegetated/mountain sample.

    Returns a list of (check, passed) tuples. These are heuristics, not proof of band order.
    """
    m = stats["mean"]
    g = lambda b: m[bands.index(b)]
    checks = []
    # NIR (B08) usually > Red (B04) over vegetated terrain
    checks.append(("NIR(B08) > RED(B04)", bool(g("B08") > g("B04"))))
    # B10 (cirrus, 1375 nm) is very dark over the surface
    checks.append(("B10 darkest optical band", bool(g("B10") <= min(m[:12]) + 1e-9)))
    # DEM should have a much larger range than reflectance
    checks.append(("DEM range > reflectance range",
                   bool((stats["max"][-1] - stats["min"][-1]) > (stats["max"][3] - stats["min"][3]))))
    return checks
