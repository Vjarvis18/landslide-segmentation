# Uploading to GitHub

```bash
unzip landslide-segmentation.zip && cd landslide-segmentation
git init -b main
git add .
git commit -m "Initial scaffold: Landslide4Sense segmentation benchmark"
# create an empty repo named landslide-segmentation on github.com (no README), then:
git remote add origin https://github.com/<your-username>/landslide-segmentation.git
git push -u origin main
```
Never commit `data/raw/` or checkpoints. They are git-ignored. Use a GitHub Release asset or
Hugging Face Hub for weights.

Tagging the release:
```bash
git tag -a v0.1.0 -m "v0.1.0: baseline, U-Net, ResU-Net, Prithvi results"
git push origin v0.1.0
```
Then create the release in GitHub → Releases → "Draft a new release" → choose `v0.1.0`.
