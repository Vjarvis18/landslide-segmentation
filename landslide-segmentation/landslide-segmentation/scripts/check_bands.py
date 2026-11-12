import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse, json
import numpy as np
from l4s.bands import L4S_BANDS, band_stats, heuristic_checks, prithvi_mapping
from l4s.io import list_patches, load_stack

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/raw/TrainData")
ap.add_argument("--n", type=int, default=200)
a = ap.parse_args()

files = list_patches(a.data)[: a.n]
X, _ = load_stack(files, with_mask=False)
print("Array shape:", X.shape)
assert X.shape[-1] == 14, "Expected 14 channels"
s = band_stats(X)
for i, b in enumerate(L4S_BANDS):
    print(f"{i:2d} {b:6s} mean={s['mean'][i]:10.4f} std={s['std'][i]:10.4f} min={s['min'][i]:10.4f} max={s['max'][i]:10.4f}")
for name, ok in heuristic_checks(s):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
idx, notes = prithvi_mapping()
print("Prithvi channel indices:", idx)
for n in notes:
    print("NOTE:", n)
os.makedirs("data/processed", exist_ok=True)
json.dump({"bands": L4S_BANDS, "mean": s["mean"].tolist(), "std": s["std"].tolist(),
           "prithvi_idx": idx, "notes": notes}, open("data/processed/band_stats.json", "w"), indent=2)
print("Saved data/processed/band_stats.json")
