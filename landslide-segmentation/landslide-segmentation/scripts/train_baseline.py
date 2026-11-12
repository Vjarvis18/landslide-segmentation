import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse, json, pickle, yaml
import numpy as np
from l4s.io import list_patches, load_patch
from l4s.features import sample_pixels, pixel_features
from l4s.metrics import scores
from l4s.models import build_baseline

ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/baseline.yaml")
cfg = yaml.safe_load(open(ap.parse_args().config))
sp = json.load(open(cfg["splits"]))
files = list_patches(cfg["data_dir"])
rng = np.random.default_rng(cfg["seed"])

Xs, ys = [], []
for i in sp["train"]:
    img, m = load_patch(files[i])
    x, y = sample_pixels(img, m, cfg["pixels_per_patch"], cfg["landslide_fraction"], rng)
    Xs.append(x); ys.append(y)
X, y = np.concatenate(Xs), np.concatenate(ys)
print("training pixels:", X.shape)
clf = build_baseline(cfg["model"], cfg[cfg["model"]])
clf.fit(X, y)

preds, gts = [], []
pred_dir = os.path.join("outputs/preds", "baseline"); os.makedirs(pred_dir, exist_ok=True)
for i in sp["val"]:
    img, m = load_patch(files[i])
    p = clf.predict(pixel_features(img)).reshape(m.shape).astype(np.uint8)
    np.save(os.path.join(pred_dir, f"{i}.npy"), p)
    preds.append(p); gts.append(m)
s = scores(np.stack(preds), np.stack(gts))
print(json.dumps(s, indent=2))
os.makedirs(cfg["out_dir"], exist_ok=True)
pickle.dump(clf, open(os.path.join(cfg["out_dir"], "model.pkl"), "wb"))
json.dump(s, open(os.path.join(cfg["out_dir"], "val_scores.json"), "w"), indent=2)
