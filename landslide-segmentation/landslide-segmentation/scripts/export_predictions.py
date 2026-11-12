import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

"""Optional: export a georeferenced prediction raster to GeoTIFF, polygons to GeoPackage and PostGIS.

Input: a single-band prediction array (.npy) plus a reference GeoTIFF supplying CRS/transform
(e.g. your Sentinel-2 stack for the Himalayan AOI). L4S patches themselves are NOT georeferenced.
"""
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--pred", required=True, help="prediction .npy (H, W), 0/1")
ap.add_argument("--ref", required=True, help="reference GeoTIFF with the same grid")
ap.add_argument("--out-tif", default="outputs/himalaya_pred.tif")
ap.add_argument("--out-gpkg", default="outputs/himalaya_pred.gpkg")
ap.add_argument("--min-area-m2", type=float, default=900.0, help="drop polygons smaller than this")
ap.add_argument("--postgis", default=None, help="postgresql://user:pass@host:5432/db (optional)")
a = ap.parse_args()

import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape

pred = np.load(a.pred).astype(np.uint8)
with rasterio.open(a.ref) as src:
    prof = src.profile.copy(); tr = src.transform; crs = src.crs
assert pred.shape == (prof["height"], prof["width"]), "prediction grid != reference grid"
prof.update(count=1, dtype="uint8", nodata=255, compress="deflate")
with rasterio.open(a.out_tif, "w", **prof) as dst:
    dst.write(pred, 1)
geoms = [shape(g) for g, v in shapes(pred, mask=pred == 1, transform=tr) if v == 1]
gdf = gpd.GeoDataFrame({"class": ["landslide"] * len(geoms)}, geometry=geoms, crs=crs)
if gdf.crs and gdf.crs.is_geographic:
    print("WARNING: geographic CRS - areas computed after reprojecting to an estimated UTM zone")
    gdf["area_m2"] = gdf.to_crs(gdf.estimate_utm_crs()).area
else:
    gdf["area_m2"] = gdf.area
gdf = gdf[gdf.area_m2 >= a.min_area_m2]
gdf.to_file(a.out_gpkg, layer="landslides", driver="GPKG")
print(f"{len(gdf)} polygons -> {a.out_gpkg}")
if a.postgis:
    from sqlalchemy import create_engine
    gdf.to_postgis("landslide_predictions", create_engine(a.postgis), if_exists="replace", index=False)
    print("written to PostGIS table landslide_predictions (add a GIST index on geometry)")
