import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

"""Score saved predictions overall and per pseudo-region (stress test)."""
import argparse, json
import numpy as np
from l4s.io import list_patches, load_patch
from l4s.metrics import scores

ap = argparse.ArgumentParser()
ap.add_argument("--pred-dir", required=True)
ap.add_argument("--data", default="data/raw/TrainData")
ap.add_argument("--splits", default="data/processed/splits.json")
a = ap.parse_args()
sp = json.load(open(a.splits)); files = list_patches(a.data)
reg = np.array(sp["pseudo_region"])
P, G, R = [], [], []
for i in sp["val"]:
    f = os.path.join(a.pred_dir, f"{i}.npy")
    if not os.path.exists(f): continue
    P.append(np.load(f)); G.append(load_patch(files[i])[1]); R.append(reg[i])
P, G, R = np.stack(P), np.stack(G), np.array(R)
res = {"overall": scores(P, G), "per_region": {int(r): scores(P[R == r], G[R == r]) for r in np.unique(R)}}
print(json.dumps({k: (v if k != "per_region" else {r: round(s["f1"], 4) for r, s in v.items()}) for k, v in res.items()}, indent=2))
json.dump(res, open(os.path.join(a.pred_dir, "eval.json"), "w"), indent=2)
