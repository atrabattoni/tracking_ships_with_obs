"""
Make temporal segmentation.
"""
# %% Imports


import obsea.tracking as tracking
import pandas as pd
import xarray as xr

# %% Parameters
obs_orientation = 77.0
obs_location = 350.0 + 1j * 50.0
cpa = 25_000.0
radius = 100_000.0
date_range = pd.date_range("2013-05-21", "2013-05-27", freq="D")
N = len(date_range) - 1

# %% Load Data
data = {}
data["a"] = xr.open_dataarray("../data/ell_a.nc")
data["r"] = xr.open_dataarray("../data/ell_r.nc")
data["rm"] = xr.open_dataarray("../data/ell_rm.nc")
data["rp"] = xr.open_dataarray("../data/ell_rp.nc")

# %% Process Data
Pd = dict(r=0.6, a=0.9)
n = dict(r=60, a=120)

ell = {
    k: tracking.detection_probability(tracking.marginal(v), Pd[k[0]])
    for k, v in data.items()
}
ell["all"] = ell["r"] * ell["a"]
ell = {k: tracking.segment(v, n[k[0]]) for k, v in ell.items()}
p = {k: tracking.ell2proba(v) for k, v in ell.items()}
dtc = {
    "vm": (p["all"] > 0.5) & (p["r"] > 0.5) & (p["rm"] > p["rp"]),
    "vp": (p["all"] > 0.5) & (p["r"] > 0.5) & (p["rm"] < p["rp"]),
}
dtc["xor"] = (p["all"] > 0.5) & (~dtc["vm"]) & (~dtc["vp"])
dtc["all"] = dtc["vm"] + dtc["vp"] + dtc["xor"]
dtc_chunk = {k: tracking.chunk(v, date_range) for k, v in dtc.items()}

dtc = xr.Dataset(dtc)

# %% Segment
segments = tracking.delimit(dtc)
segments = pd.DataFrame(segments, columns=["starttime", "cpatime", "endtime"])

# %% Save
dtc.to_netcdf("../data/dtc.nc")
segments.to_csv("../data/segments.csv", index=False)
