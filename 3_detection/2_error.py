# %% Import libs

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import obsea

amax = 45
rmax = 5000
vmax = 3

# %% Prepare data

# Load peaks
a = xr.open_dataarray("../data/a.nc")
r = xr.open_dataarray("../data/r.nc")
v = xr.open_dataarray("../data/v.nc")

# Load tracks
tracks = pd.read_pickle("../data/tracks.pkl")
# tracks = tracks["linestring"]
tracks = tracks.apply(lambda xarr: xarr.interp_like(a))
tracks = xr.Dataset(tracks.to_dict())
name_dict = {key: str(k) for k, key in enumerate(tracks.keys())}
tracks = tracks.rename(name_dict)
obs_orientation = 77.0
tracks *= np.exp(1j*np.deg2rad(obs_orientation))

# Convert tracks into srange and speed and aling everybody
atrack = np.rad2deg(np.arctan2(tracks.real, tracks.imag)) % 360
rtrack = np.abs(tracks) 
vtrack = np.abs(tracks).diff("time") / 60.0

a, r, v, atrack, rtrack, vtrack = xr.align(a, r, v, atrack, rtrack, vtrack)

# %% Global Error

# Error
diff_ag = (a - atrack).to_array("track")
diff_ag = (diff_ag + 180) % 360 - 180
diff_rg = (r - rtrack).to_array("track")
diff_vg = (v - vtrack).to_array("track")

# Remove times without data
diff_ag = diff_ag.dropna("time", "all")
diff_rg = diff_rg.dropna("time", "all")
diff_vg = diff_vg.dropna("time", "all")

# Chose minimum error
index = np.abs(diff_ag).argmin("track")
diff_ag = diff_ag[index]
index = np.abs(diff_rg).argmin("track")
diff_rg = diff_rg[index]
index = np.abs(diff_vg).argmin("track")
diff_vg = diff_vg[index]

# Filter out too big errors
cond_ag = (np.abs(diff_ag) < amax)
diff_truncated_ag = xr.where(cond_ag, diff_ag, np.nan)
cond_g = (np.abs(diff_rg) < rmax) & (np.abs(diff_vg) < vmax)
diff_truncated_rg = xr.where(cond_g, diff_rg, np.nan)
diff_truncated_vg = xr.where(cond_g, diff_vg, np.nan)

# Compute SMAD
smad_ag = 1.4826 * np.abs(diff_truncated_ag).median()
smad_rg = 1.4826 * np.abs(diff_truncated_rg).median()
smad_vg = 1.4826 * np.abs(diff_truncated_vg).median()

# %% Per ship error

# Error
diff_a = (a - atrack)
diff_a = (diff_a + 180) % 360 - 180
diff_r = (r - rtrack)
diff_v = (v - vtrack)

# Filter out to big errors
cond_a = (np.abs(diff_a) < amax)
diff_truncated_a = xr.where(cond_a, diff_a, np.nan)
cond_rv = (np.abs(diff_r) < rmax) & (np.abs(diff_v) < vmax)
diff_truncated_r = xr.where(cond_rv, diff_r, np.nan)
diff_truncated_v = xr.where(cond_rv, diff_v, np.nan)

# Compute SMAD
smad_a = 1.4826 * np.abs(diff_truncated_a).median("time")
smad_r = 1.4826 * np.abs(diff_truncated_r).median("time")
smad_v = 1.4826 * np.abs(diff_truncated_v).median("time")

# Count values
count_a = cond_a.apply(np.count_nonzero)
count_rv = cond_rv.apply(np.count_nonzero)

# Format
smad_a = [smad_a[key].values.item() for key in smad_a]
smad_r = [smad_r[key].values.item() for key in smad_r]
smad_v = [smad_v[key].values.item() for key in smad_v]
count_a = [count_a[key].values.item() for key in count_a]
count_rv = [count_rv[key].values.item() for key in count_rv]

df = pd.DataFrame()
df["Na"] = count_a
df["Heading"] = np.round(smad_a, 1)
df["Nr"] = count_rv
df["Distance"] = np.round(smad_r, 0)
df["Speed"] = np.round(smad_v, 2)
df.index += 1
df.to_excel("../data/detection_errors.xlsx")



df = pd.DataFrame({
    "Heading": [smad_ag.values.item()],
    "Distance": [smad_rg.values.item()],
    "Speed": [smad_vg.values.item()],
})
df.to_excel("../data/detection_general_errors.xlsx")
