import pickle

import numpy as np
from obspy import read, read_inventory
import obsea
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Dates
date_range = pd.date_range("2013-05-21", "2013-05-27", freq="2D")
N = len(date_range) - 1

# Load detections
a = xr.open_dataarray("a.nc")
r = xr.open_dataarray("r.nc")
v = xr.open_dataarray("v.nc")
t = pd.to_datetime(a["time"].values, unit="s")

# Load tracks
tracks = pd.read_pickle("tracks.pkl")
xtracks = tracks["linestring"].apply(obsea.track2xarr)
xtracks = xtracks.apply(lambda xarr: xarr.interp_like(a["time"]))
atracks = xtracks.apply(lambda xarr: (np.rad2deg(
    np.arctan2(xarr.real, xarr.imag)) - 77) % 360)
rtracks = xtracks.apply(lambda xarr: np.abs(xarr))

# # Load loglik
# ell_a = xr.open_dataarray("ell_a.nc")
# ell_r = xr.open_dataarray("ell_r.nc")

# Load segments
with open("dtc.pkl", "rb") as file:
    dtc = pickle.load(file)
with open("segments.pkl", "rb") as file:
    segments = pickle.load(file)


# Plot
plt.style.use("figures.mplstyle")
blank = 0.2
bar = 0.07
fig, axes = plt.subplots(nrows=11, figsize=(7.1, 5),
                         gridspec_kw=dict(
    hspace=0.0, wspace=0.0,
    left=0.06, right=0.98,
    bottom=0.04, top=0.99,
    height_ratios=[1, bar, 1, blank, 1, bar, 1, blank, 1, bar, 1]
))
for i in range(N):
    ax = axes[4*i]
    # ax.pcolormesh(t, ell_a["azimuth"], np.log(ell_a).T,
    #               cmap="binary", vmin=0, vmax=20, rasterized=True)
    ax.scatter(t, a, marker="s", s=4, fc="none", ec="C4", linewidth=0.5)
    for atrack in atracks:
        ax.plot(t, atrack, c="black", ls="-.")
    for segment in segments:
        ax.axvline(segment[0], c="C2", ls=":")
        ax.axvline(segment[1], c="black", ls=":")
        ax.axvline(segment[2], c="C3", ls=":")
    ax.set_xlim(date_range[i], date_range[i + 1])
    ax.tick_params(bottom=False, labelbottom=False)
    ax.set_ylim(0, 360)
    ax.set_yticks([0, 90, 180, 270, 360])
    ax.set_ylabel("Azimuth [°]")

    ax = axes[4*i + 1]
    ax.fill_between(
        dtc["xor"]["time"], 0, 1 * dtc["xor"],
        facecolor="gray", lw=0, edgecolor="none", step="mid")
    ax.fill_between(
        dtc["vm"]["time"], 0, 1 * dtc["vm"],
        facecolor="C2", lw=0, edgecolor="none", step="mid")
    ax.fill_between(
        dtc["vp"]["time"], 0, 1 * dtc["vp"],
        facecolor="C3", lw=0, edgecolor="none", step="mid")
    ax.tick_params(bottom=False, labelbottom=False,
                   labelleft=False, left=False)
    ax.set_xlim(date_range[i], date_range[i + 1])
    for segment in segments:
        ax.axvline(segment[0], c="C2", ls=":")
        ax.axvline(segment[1], c="black", ls=":")
        ax.axvline(segment[2], c="C3", ls=":")

    ax = axes[4*i + 2]
    # ax.pcolormesh(t, ell_r["distance"]/1000, np.log(ell_r).T,
    #               cmap="binary", vmin=0, vmax=100, rasterized=True)
    ax.scatter(t, r/1000, marker="s", s=4, fc="none", c=v*1.943844,
               linewidth=0.5, cmap="cet_diverging_gwr_55_95_c38")
    for rtrack in rtracks:
        ax.plot(t, rtrack/1000, c="black", ls="-.")
    for segment in segments:
        ax.axvline(segment[0], c="C2", ls=":")
        ax.axvline(segment[1], c="black", ls=":")
        ax.axvline(segment[2], c="C3", ls=":")
    ax.annotate(date_range[i].strftime("%d/%m"), (3, 3),
                xycoords='axes points', color="black")
    ax.set_xlim(date_range[i], date_range[i + 1])
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax.tick_params(labelbottom=False)
    ax.set_ylim(0, 50)
    ax.set_yticks(np.arange(0, 60, 10))
    ax.set_ylabel("Distance [km]")

axes[3].axis("off")
axes[7].axis("off")

ax.tick_params(labelbottom=True)
ax.set_xticklabels(
    ["00:00", "06:00", "12:00", "18:00",
     "00:00", "06:00", "12:00", "18:00",
     "00:00"])
fig.savefig("figs/detection.pdf")
