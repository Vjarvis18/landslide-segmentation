import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

"""Write 6-band (Prithvi order) .npy patches + masks for a TerraTorch datamodule."""
import argparse, json
import numpy as np
from l4s.bands import prithvi_mapping
from l4s.io import list_patches, load_patch

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/raw/TrainData")
ap.add_argument("--splits", default="data/processed/splits.json")
ap.add_argument("--out", default="data/processed/prithvi")
a = ap.parse_args()
idx, notes = prithvi_mapping()
for n in notes: print("NOTE:", n)
sp = json.load(open(a.splits)); files = list_patches(a.data)
for part in ["train", "val"]:
    for sub in ["img", "mask"]: os.makedirs(os.path.join(a.out, part, sub), exist_ok=True)
    for i in sp[part]:
        img, m = load_patch(files[i])
        np.save(os.path.join(a.out, part, "img", f"{i}.npy"), img[..., idx].transpose(2, 0, 1))
        np.save(os.path.join(a.out, part, "mask", f"{i}.npy"), m)
print("done ->", a.out, "| NOTE: verify L4S reflectance scaling vs HLS before fine-tuning")
