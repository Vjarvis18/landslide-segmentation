import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse, json, yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from l4s.io import list_patches, load_stack, normalise
from l4s.metrics import scores
from l4s.models import build_segmentation_model, dice_bce_loss

ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/unet.yaml")
cfg = yaml.safe_load(open(ap.parse_args().config))
torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
sp = json.load(open(cfg["splits"]))
files = list_patches(cfg["data_dir"])
X, Y = load_stack(files)
mean = X[sp["train"]].mean((0, 1, 2)); std = X[sp["train"]].std((0, 1, 2))  # train-only stats
X = normalise(X, mean, std)
to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a))
Xt = to_t(X.transpose(0, 3, 1, 2)).float(); Yt = to_t(Y[:, None]).float()
tr = DataLoader(TensorDataset(Xt[sp["train"]], Yt[sp["train"]]), batch_size=cfg["batch_size"], shuffle=True)
va = DataLoader(TensorDataset(Xt[sp["val"]], Yt[sp["val"]]), batch_size=cfg["batch_size"])

dev = "cuda" if torch.cuda.is_available() else "cpu"
model = build_segmentation_model(cfg["arch"], cfg["in_channels"], cfg.get("encoder"), cfg.get("encoder_weights")).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
loss_fn = dice_bce_loss()
best, os_out = -1, cfg["out_dir"]; os.makedirs(os_out, exist_ok=True)

def flip(x, y):
    if cfg.get("augment") and torch.rand(1) < 0.5: x, y = x.flip(-1), y.flip(-1)
    if cfg.get("augment") and torch.rand(1) < 0.5: x, y = x.flip(-2), y.flip(-2)
    return x, y

for ep in range(cfg["epochs"]):
    model.train()
    for xb, yb in tr:
        xb, yb = flip(xb.to(dev), yb.to(dev))
        opt.zero_grad(); l = loss_fn(model(xb), yb); l.backward(); opt.step()
    model.eval(); P, G = [], []
    with torch.no_grad():
        for xb, yb in va:
            P.append((torch.sigmoid(model(xb.to(dev))) > 0.5).cpu().numpy()); G.append(yb.numpy())
    s = scores(np.concatenate(P), np.concatenate(G))
    print(f"epoch {ep+1:3d} loss={l.item():.4f} valF1={s['f1']:.4f} valIoU={s['iou']:.4f}")
    if s["f1"] > best:
        best = s["f1"]; torch.save(model.state_dict(), os.path.join(os_out, "best.pth"))
        json.dump(s, open(os.path.join(os_out, "val_scores.json"), "w"), indent=2)
        pd = os.path.join("outputs/preds", cfg["arch"]); os.makedirs(pd, exist_ok=True)
        for j, i in enumerate(sp["val"]):
            np.save(os.path.join(pd, f"{i}.npy"), np.concatenate(P)[j, 0].astype(np.uint8))
json.dump({"mean": mean.tolist(), "std": std.tolist()}, open(os.path.join(os_out, "norm.json"), "w"))
print("best val F1:", best)
