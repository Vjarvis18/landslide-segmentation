# PROJECT_PLAN — landslide-segmentation (5 weeks, 8–10 h/week)

## Goal
A reproducible benchmark of classical ML vs CNNs vs a geospatial foundation model for landslide
segmentation on Landslide4Sense, with an honest spatial-generalisation stress test and a GIS-ready export.

## Feasibility verdict
**Feasible with modifications.** Three dataset constraints change the design:
| Constraint | Consequence | Mitigation |
|---|---|---|
| 12 optical channels and one S2 band (B8A or B10) is missing | Prithvi's 6-band HLS input cannot be matched exactly | Verify order (`check_bands.py`), substitute B08 for B8A if needed, and document it |
| Masks only for the training split on Zenodo | No official labelled validation/test set | Cluster-held-out validation from training patches |
| No region IDs | True leave-one-region-out is not possible | Pseudo-regions via k-means on patch statistics |

## Week-by-week
| Week | Hours | Deliverables | Done when |
|---|---|---|---|
| 1 | 8–10 | Repo, env, data download, `check_bands.py`, `make_splits.py`, EDA notebook (class balance, band stats) | `splits.json` + `band_stats.json` committed (not data) |
| 2 | 8–10 | RF + LightGBM per-pixel baseline, metric pipeline, first results table | `outputs/baseline/val_scores.json` |
| 3 | 8–10 | U-Net and ResU-Net (ResNet-34) training, augmentation, threshold tuning on val | Best checkpoints + val F1/IoU |
| 4 | 8–10 | Prithvi-EO-2.0 fine-tune via TerraTorch (`prepare_prithvi.py`), pseudo-region stress test for all models | Comparison table incl. per-region F1 spread |
| 5 | 8–10 | Optional Himalayan AOI inference → GeoTIFF/GeoPackage/PostGIS, README figures, limitations, release `v0.1.0` | GitHub release tagged |

## Evaluation protocol
- Metrics: landslide-class Precision, Recall, F1, IoU, and mIoU (pixel-wise, aggregated over the split).
- Primary: cluster-held-out validation (2 of 8 clusters, seed 42). Report mean ± std over 3 seeds if time allows.
- Stress test: leave-one-pseudo-region-out, or per-region scores. Report the F1 range, not just the mean.
- Normalisation stats are computed **on training patches only**.

## Risks
- Severe class imbalance (landslide pixels are a small share). Use Dice+BCE, stratified pixel sampling and per-class metrics. Never report accuracy alone.
- Reflectance scaling of L4S differs from HLS. Check value ranges before feeding them to Prithvi.
- The Himalayan run is out-of-distribution (different sensor dates, terrain and illumination). Present it qualitatively unless you have reference inventories.
- Prithvi-EO-2.0 300M needs a GPU with about 16 GB or more. Fall back to Colab/Kaggle, or freeze the backbone.

## Release checklist (v0.1.0)
- [ ] README results table + figures
- [ ] `environment.yml` pinned versions (`conda env export --from-history`)
- [ ] CI green (`tests.yml`)
- [ ] CITATION / dataset licence noted
- [ ] `git tag -a v0.1.0 -m "First benchmark release" && git push origin v0.1.0`
