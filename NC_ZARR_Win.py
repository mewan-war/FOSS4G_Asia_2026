# -*- coding: utf-8 -*-

# ==================================================
# PROJ SETUP (MUST BE FIRST)
# ==================================================
import os
import pyproj
from pyproj import CRS

os.environ["PROJ_LIB"]  = r"C:\Users\Lenovo\anaconda3\envs\INSAT_GEO\Library\share\proj"
os.environ["PROJ_DATA"] = r"C:\Users\Lenovo\anaconda3\envs\INSAT_GEO\Library\share\proj"

print("PROJ version:", pyproj.proj_version_str)
print("PROJ data dir:", pyproj.datadir.get_data_dir())
print(CRS.from_epsg(4326))
print("\n**************\n")

# ==================================================
# IMPORTS
# ==================================================
import xarray as xr
import geopandas as gpd
import rioxarray  # noqa
import zarr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ==================================================
# PATHS
# ==================================================
nc_path  = r"C:\Users\Mewan\FWI_2014_2018_combined.nc"
shp_path = r"C:\Users\Mewan\India_Country_Boundary.shp"
zarr_out = r"C:\zarr\fwi_india_2016.zarr"

# ==================================================
# LOAD NETCDF
# ==================================================
print("Opening FWI NetCDF...")

ds = xr.open_dataset(
    nc_path,
    decode_times=True,
    use_cftime=False
)

# ==================================================
# TIME FIX (CRITICAL)
# ==================================================
if not np.issubdtype(ds.time.dtype, np.datetime64):
    print("Fixing integer time axis ? datetime64")

    # Try CF decoding manually
    ds = xr.decode_cf(ds, use_cftime=False)

# Final guarantee
if not np.issubdtype(ds.time.dtype, np.datetime64):
    print("Forcing daily datetime index as fallback")

    time_index = pd.date_range(
        start="2016-01-01",
        periods=ds.sizes["time"],
        freq="D"
    )

    ds = ds.assign_coords(time=("time", time_index))

print("Final time dtype:", ds.time.dtype)


print("Original variables:", list(ds.data_vars))
print("Time dtype:", ds.time.dtype)

# Select only FWI
fwi = ds["FWI"]

# ==================================================
# LOAD INDIA SHAPEFILE
# ==================================================
India = gpd.read_file(shp_path).to_crs("EPSG:4326")

# ==================================================
# SPATIAL METADATA (EPSG:4326)
# ==================================================
fwi = fwi.rio.write_crs("EPSG:4326", inplace=True)

fwi = fwi.rio.set_spatial_dims(
    x_dim="longitude",
    y_dim="latitude",
    inplace=True
)

# ==================================================
# CLIP TO INDIA (TIME PRESERVED)
# ==================================================
print("Clipping FWI cube to India...")
fwi_india = fwi.rio.clip(
    India.geometry,
    India.crs,
    drop=True
)

print("FWI India dims:", fwi_india.dims)

# ==================================================
# TIME SAFETY CHECK (CRITICAL FOR PYGEOAPI)
# ==================================================
assert np.issubdtype(
    fwi_india.time.dtype, np.datetime64
), "? Time axis is NOT datetime64!"

# ==================================================
# CHUNKING (OPTIMAL FOR COVERAGE ACCESS)
# ==================================================
spatial_chunk = 100

fwi_india = fwi_india.chunk({
    "time": 1,
    "latitude": spatial_chunk,
    "longitude": spatial_chunk,
})

# ==================================================
# DATAARRAY ? DATASET
# ==================================================
ds_base = fwi_india.to_dataset(name="FWI")

# Drop stray CRS variable if present (avoids pygeoapi noise)
ds_base = ds_base.drop_vars("crs", errors="ignore")

# ==================================================
# ZARR v3 WRITE (RAW CUBE)
# ==================================================
compressor = zarr.codecs.ZstdCodec(level=1)

encoding = {
    "FWI": {
        "dtype": "float32",
        "chunks": (1, spatial_chunk, spatial_chunk),
        "compressors": [compressor],
    }
}

print("Writing Zarr v3 (EPSG:4326, raw cube)...")

ds_base.to_zarr(
    zarr_out,
    mode="w",
    zarr_format=3,
    consolidated=False,   # correct for Zarr v3
    encoding=encoding,
)

# ==================================================
# SANITY CHECK
# ==================================================
print("Opening written Zarr...")
ds_check = xr.open_zarr(zarr_out, consolidated=False)
print(ds_check)

print("Time dtype:", ds_check.time.dtype)
print("Time values:", ds_check.time.values[:3])

# ==================================================
# QUICK VISUAL TEST
# ==================================================
fwi_day10 = ds_check["FWI"].isel(time=9)

plt.figure(figsize=(7, 5))
fwi_day10.plot(cmap="Oranges", robust=True)
plt.title("FWI India - Day 10 (Zarr v3, raw cube)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

print("? FWI Zarr v3 raw cube is READY for pygeoapi.")
