import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse, json
from l4s.io import list_patches, load_stack
from l4s.splits import patch_descriptors, kmeans, cluster_split

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/raw/TrainData")
ap.add_argument("--k", type=int, default=8, help="number of pseudo-regions")
ap.add_argument("--val-clusters", type=int, default=2)
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

files = list_patches(a.data)
X, Y = load_stack(files)
lab = kmeans(patch_descriptors(X), a.k, seed=a.seed)
tr, va, held = cluster_split(lab, a.val_clusters, seed=a.seed)
out = {"files": [os.path.basename(f) for f in files], "pseudo_region": lab.tolist(),
       "train": tr.tolist(), "val": va.tolist(), "held_out_clusters": held, "k": a.k, "seed": a.seed}
os.makedirs("data/processed", exist_ok=True)
json.dump(out, open("data/processed/splits.json", "w"))
print(f"train={len(tr)} val={len(va)} held-out clusters={held}")
if Y is not None:
    print(f"landslide pixel share train={Y[tr].mean():.4f} val={Y[va].mean():.4f}")
