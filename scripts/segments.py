# %% Imports

import pickle

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import obsea
import pandas as pd
import xarray as xr
from obspy.clients.fdsn import Client
from pandas import to_datetime

import utils



# %% Load Data

obs_orientation = 77.0
obs_location = 350.0 + 1j*50.0
cpa = 25_000.0
radius = 100_000.0
date_range = pd.date_range("2013-05-21", "2013-05-27", freq="D")
N = len(date_range) - 1

data = {}
data["a"] = xr.open_dataarray("ell_a.nc")
data["r"] = xr.open_dataarray("ell_r.nc")
data["rm"] = xr.open_dataarray("ell_rm.nc")
data["rp"] = xr.open_dataarray("ell_rp.nc")

data["a"]["time"] = to_datetime(data["a"]["time"].values, unit="s")
data["r"]["time"] = to_datetime(data["r"]["time"].values, unit="s")
data["rm"]["time"] = to_datetime(data["r"]["time"].values, unit="s")
data["rp"]["time"] = to_datetime(data["r"]["time"].values, unit="s")


# %% Process Data

Pd = dict(r=0.6, a=0.9)
n = dict(r=60, a=120)

ell = {k: utils.detection_probability(utils.marginal(v), Pd[k[0]])
       for k, v in data.items()}
ell["all"] = ell["r"] * ell["a"]
ell = {k: utils.segment(v, n[k[0]])
       for k, v in ell.items()}
p = {k: utils.ell2proba(v) for k, v in ell.items()}
dtc = {
    "vm": (p["all"] > 0.5) & (p["r"] > 0.5) & (p["rm"] > p["rp"]),
    "vp": (p["all"] > 0.5) & (p["r"] > 0.5) & (p["rm"] < p["rp"]),
}
dtc["xor"] = (p["all"] > 0.5) & (~dtc["vm"]) & (~dtc["vp"])
dtc["all"] = dtc["vm"] + dtc["vp"] + dtc["xor"]
dtc_chunk = {k: utils.chunk(v, date_range) for k, v in dtc.items()}


# %% Segment

segments = utils.delimit(dtc)

with open("dtc.pkl", "wb") as file:
    pickle.dump(dtc, file)
with open("segments.pkl", "wb") as file:
    pickle.dump(segments, file)


# %% Plot

plt.style.use("figures.mplstyle")
fig, axes = plt.subplots(nrows=N, dpi=300, figsize=(5.3, 8))
for i in range(N):
    ax = axes[i]
    ax_right = ax.twinx()
    # # Tracks
    # for track in local_tracks:
    #     xtrack = obsea.track2xarr(track)
    #     xtrack -= obs_location
    #     xtrack *= np.exp(1j*np.deg2rad(obs_orientation))
    #     xtrack["time"] = pd.to_datetime(xtrack["time"].values, unit="s")
    #     xtrack = xtrack.interp_like(dtc_chunk["xor"][i])
    #     ax.plot(xtrack["time"], np.rad2deg(
    #         np.arctan2(xtrack.real, xtrack.imag)) % 360,
    #         color="black", ls=":")
    #     ax_right.plot(
    #         xtrack["time"], np.abs(xtrack) / 1000,
    #         color="black", ls="--")
    # Detection segments
    ax.fill_between(
        dtc_chunk["xor"][i]["time"], 0, 360 * dtc_chunk["xor"][i],
        facecolor="gray", alpha=0.2, lw=0, edgecolor="none", step="mid")
    ax.fill_between(
        dtc_chunk["vm"][i]["time"], 0, 360 * dtc_chunk["vm"][i],
        facecolor="C2", alpha=0.2, lw=0, edgecolor="none", step="mid")
    ax.fill_between(
        dtc_chunk["vp"][i]["time"], 0, 360 * dtc_chunk["vp"][i],
        facecolor="C3", alpha=0.2, lw=0, edgecolor="none", step="mid")
    # Limits
    for segment in segments:
        ax.axvline(segment[0], c="C2", lw=1)
        ax.axvline(segment[1], c="black", lw=1)
        ax.axvline(segment[2], c="C3", lw=1)
    # Formating
    ax.set_ylabel("Azimuth (°)")
    ax.set_yticks(np.arange(0, 361, 45))
    ax.set_ylim((0, 360))

    ax_right.set_ylabel("Distance (km)")
    ax_right.set_ylim([0, 50])
    ax_right.set_yticks(np.arange(0, 50 + 10, 10))

    ax.set_xlim(date_range[i], date_range[i + 1])
    ax.tick_params(labelbottom=False)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))

axes[-1].set_xlabel("Time")
axes[-1].tick_params(labelbottom=True)
axes[-1].set_xticklabels(
    ["00:00", "06:00", "12:00", "18:00", "00:00"])
fig.tight_layout()
fig.savefig("segments.pdf")

# %%
