# landslide-segmentation

Pixel-wise landslide segmentation on the **Landslide4Sense (L4S)** benchmark, comparing:

| # | Model | Type | Notes |
|---|-------|------|-------|
| 1 | Random Forest / LightGBM | Per-pixel baseline | Bands + NDVI/NDWI + slope/DEM features |
| 2 | U-Net | CNN from scratch | 14-channel input |
| 3 | ResU-Net | CNN, ResNet-34 encoder | 14-channel input, first conv adapted |
| 4 | Prithvi-EO-2.0 (TerraTorch) | Geospatial foundation model fine-tune | 6-band HLS mapping. See the band caveat |

Plus a **pseudo-region stress test** (spatial generalisation proxy) and an optional **Himalayan inference
run** exported to GeoTIFF, GeoPackage and PostGIS. Release tag: `v0.1.0`.

## Dataset caveats (read first)
1. **Band gap:** the 12 optical L4S channels do not cover every Sentinel-2 band (B8A or B10 is missing).
   Run `python scripts/check_bands.py` and confirm the order against the L4S paper/README *before*
   building the Prithvi mapping (Prithvi-EO-2.0 expects B02, B03, B04, B8A, B11, B12).
   If B8A is absent, B08 is used as a NIR proxy, which is a documented approximation.
2. **No public validation/test masks:** the Zenodo release ships masks for training only. The validation
   set is carved from **spatial clusters of training patches** (`scripts/make_splits.py`), not from random
   patches, to limit leakage between neighbouring/overlapping patches.
3. **No region labels:** patches are not tagged by study area, so the stress test uses **pseudo-regions**
   (k-means clusters of per-patch statistics). Treat it as a proxy, not a true geographic hold-out.

## Quick start
```bash
conda env create -f environment.yml && conda activate l4s
python scripts/download_data.py                  # Zenodo instructions + file check
python scripts/check_bands.py --data data/raw/TrainData
python scripts/make_splits.py --data data/raw/TrainData --k 8 --val-clusters 2
python scripts/train_baseline.py --config configs/baseline.yaml
python scripts/train_unet.py --config configs/unet.yaml
python scripts/train_unet.py --config configs/resunet.yaml
terratorch fit -c configs/prithvi_terratorch.yaml
python scripts/evaluate.py --pred-dir outputs/preds/unet --data data/raw/TrainData --splits data/processed/splits.json
python scripts/export_predictions.py --help      # optional Himalayan run -> GeoTIFF/GPKG/PostGIS
```

## Layout
```
configs/   YAML configs per model
scripts/   CLI entry points
src/l4s/   library code (io, features, splits, metrics, models)
tests/     numpy-only unit tests (CI)
docs/      PROJECT_PLAN.md, GITHUB_UPLOAD.md
outputs/   (git-ignored) checkpoints, predictions, figures
```

## Metrics
Landslide-class Precision, Recall, F1, IoU, plus mIoU. Reported on the cluster-held-out validation split
and per pseudo-region.

## Citation
Ghorbanzadeh, O. et al. (2022). The Outcome of the 2022 Landslide4Sense Competition: Advanced Landslide
Detection From Multisource Satellite Imagery. IEEE JSTARS. Dataset hosted on Zenodo.
