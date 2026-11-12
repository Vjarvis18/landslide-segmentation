"""Binary segmentation metrics for the landslide class."""
import numpy as np


def confusion(pred, target):
    p = np.asarray(pred).astype(bool).ravel()
    t = np.asarray(target).astype(bool).ravel()
    tp = int(np.sum(p & t)); fp = int(np.sum(p & ~t))
    fn = int(np.sum(~p & t)); tn = int(np.sum(~p & ~t))
    return tp, fp, fn, tn


def scores(pred, target, eps=1e-9):
    tp, fp, fn, tn = confusion(pred, target)
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    iou1 = tp / (tp + fp + fn + eps)
    iou0 = tn / (tn + fp + fn + eps)
    return {"precision": prec, "recall": rec, "f1": f1, "iou": iou1,
            "miou": (iou0 + iou1) / 2, "tp": tp, "fp": fp, "fn": fn, "tn": tn}
