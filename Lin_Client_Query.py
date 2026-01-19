
# -*- coding: utf-8 -*-

import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# ==================================================
# CONFIG
# ==================================================
URL_ROOT = "http://localhost:5000"
#URL_ROOT = "http://10.2.141.73:5000"
COLLECTION = "fwi_india"

BBOX = "78.25,17.25,78.75,17.75"
REQUESTED_TIME = "2016-03-15T00:00:00Z"
FWI_TIMERANGE = "2016-03-01/2016-03-31"

# ==================================================
# 1. REQUEST SINGLE-DAY EDR CUBE
# ==================================================
cube_url = f"{URL_ROOT}/collections/{COLLECTION}/cube"

params = {
    "bbox": BBOX,
    "datetime": REQUESTED_TIME,
    "parameter-name": "FWI",
    "f": "json"
}

response = requests.get(cube_url, params=params)
response.raise_for_status()
cube = response.json()

# ==================================================
# 2. DETECT AXES (ROBUST)
# ==================================================
axes = cube["domain"]["axes"]

x_name = "longitude" if "longitude" in axes else "x"
y_name = "latitude" if "latitude" in axes else "y"

x_axis = axes[x_name]
y_axis = axes[y_name]

# time may be collapsed
if "time" in axes:
    t_axis = axes["time"]
    time_vals = pd.date_range(
        start=t_axis["start"],
        periods=t_axis["num"],
        freq="D"
    )
else:
    time_vals = [pd.to_datetime(REQUESTED_TIME)]

# ==================================================
# 3. BUILD AXIS VALUES
# ==================================================
x_vals = np.linspace(x_axis["start"], x_axis["stop"], x_axis["num"])
y_vals = np.linspace(y_axis["start"], y_axis["stop"], y_axis["num"])

# ==================================================
# 4. EXTRACT VALUES
# ==================================================
fwi_vals = cube["ranges"]["FWI"]["values"]

# ==================================================
# 5. RECONSTRUCT GRID (SAFE)
# ==================================================
records = []
idx = 0
n_vals = len(fwi_vals)

for t in time_vals:
    for y in y_vals:
        for x in x_vals:
            if idx >= n_vals:
                break
            records.append({
                "time": t,
                "longitude": x,
                "latitude": y,
                "FWI": fwi_vals[idx]
            })
            idx += 1

df = pd.DataFrame(records)

gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.longitude, df.latitude),
    crs="EPSG:4326"
)

print("GeoDataFrame preview (single day):")
print(gdf.head())

# ==================================================
# 6. PLOT SINGLE-DAY MAP
# ==================================================
fig, ax = plt.subplots(figsize=(7, 6))
gdf.plot(column="FWI", cmap="Oranges", markersize=80, legend=True, ax=ax)
ax.set_title(f"Fire Weather Index (FWI) - {pd.to_datetime(REQUESTED_TIME).date()}")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.tight_layout()
plt.show()

# ==================================================
# 7. MULTI-DAY EDR CUBE (PRISM-LIKE FEATURE)
# ==================================================
params_range = {
    "bbox": BBOX,
    "datetime": FWI_TIMERANGE,
    "parameter-name": "FWI",
    "f": "json"
}

response_range = requests.get(cube_url, params=params_range)
response_range.raise_for_status()
cube_range = response_range.json()

axes_r = cube_range["domain"]["axes"]

x_name = "longitude" if "longitude" in axes_r else "x"
y_name = "latitude" if "latitude" in axes_r else "y"

x_axis = axes_r[x_name]
y_axis = axes_r[y_name]

x_vals = np.linspace(x_axis["start"], x_axis["stop"], x_axis["num"])
y_vals = np.linspace(y_axis["start"], y_axis["stop"], y_axis["num"])

# time axis may or may not exist
if "time" in axes_r:
    t_axis = axes_r["time"]
    time_vals = pd.date_range(
        start=t_axis["start"],
        periods=t_axis["num"],
        freq="D"
    )
else:
    start, end = FWI_TIMERANGE.split("/")
    time_vals = pd.date_range(start=start, end=end, freq="D")

fwi_vals = cube_range["ranges"]["FWI"]["values"]

# ==================================================
# 8. RECONSTRUCT RANGE CUBE (SAFE)
# ==================================================
records = []
idx = 0
n_vals = len(fwi_vals)
nx = len(x_vals)
ny = len(y_vals)
n_space = nx * ny
n_time = max(1, n_vals // n_space)

time_vals = time_vals[:n_time]

for t in time_vals:
    for y in y_vals:
        for x in x_vals:
            if idx >= n_vals:
                break
            records.append({
                "time": t,
                "longitude": x,
                "latitude": y,
                "FWI": fwi_vals[idx]
            })
            idx += 1

df_cube = pd.DataFrame(records)

# ==================================================
# 9. DAILY MEAN FWI TIME SERIES
# ==================================================
daily_mean_fwi = (
    df_cube
    .groupby("time")["FWI"]
    .mean()
    .reset_index()
)

print("\nDaily mean FWI over bbox:")
print(daily_mean_fwi.head())

# ==================================================
# 10. PLOT TIME SERIES
# ==================================================
plt.figure(figsize=(8, 4))
plt.plot(daily_mean_fwi["time"], daily_mean_fwi["FWI"], marker="o")
plt.title("Daily Mean Fire Weather Index (FWI): Hyderabad neighbourhood")
plt.xlabel("Date")
plt.ylabel("FWI")
plt.grid(True)
plt.tight_layout()
plt.show()
