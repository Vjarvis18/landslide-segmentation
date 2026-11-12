"""Reading Landslide4Sense .h5 patches (keys: 'img' (128,128,14), 'mask' (128,128))."""
import glob
import os
import numpy as np


def list_patches(data_dir, sub="img"):
    pat = os.path.join(data_dir, sub, "*.h5")
    files = sorted(glob.glob(pat))
    if not files:
        raise FileNotFoundError(f"No .h5 files in {pat}")
    return files


def mask_path_for(img_path):
    d, f = os.path.split(img_path)
    return os.path.join(os.path.dirname(d), "mask", f.replace("image", "mask"))


def read_h5(path, key):
    import h5py
    with h5py.File(path, "r") as h:
        return np.asarray(h[key])


def load_patch(img_path, with_mask=True):
    img = read_h5(img_path, "img").astype(np.float32)
    mask = None
    if with_mask:
        mp = mask_path_for(img_path)
        mask = read_h5(mp, "mask").astype(np.uint8) if os.path.exists(mp) else None
    return img, mask


def load_stack(files, with_mask=True):
    imgs, masks = [], []
    for f in files:
        i, m = load_patch(f, with_mask)
        imgs.append(i)
        masks.append(m)
    X = np.stack(imgs)
    Y = np.stack(masks) if with_mask and all(m is not None for m in masks) else None
    return X, Y


def normalise(X, mean, std, eps=1e-6):
    return (X - mean) / (std + eps)
