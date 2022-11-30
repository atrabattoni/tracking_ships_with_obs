"""
Plot figure 7.
"""

# %% Libs
from glob import glob

import colorcet
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import obsea
import pandas as pd
import xarray as xr
from obspy import read, read_inventory

# %% Dates
date_range = pd.date_range("2013-05-21", "2013-05-27", freq="2D")
N = len(date_range) - 1

# %% Load detections
a = xr.open_dataarray("../data/a.nc")
r = xr.open_dataarray("../data/r.nc")
v = xr.open_dataarray("../data/v.nc")
t = pd.to_datetime(a["time"].values, unit="s")

# %% Load tracks
fnames = sorted(glob("../data/track_*.nc"))
tracks = pd.Series([obsea.read_complex(fname) for fname in fnames])
tracks = tracks.apply(lambda xarr: xarr.interp_like(a["time"]))
atracks = tracks.apply(
    lambda xarr: (np.rad2deg(np.arctan2(xarr.real, xarr.imag)) - 77) % 360
)
rtracks = tracks.apply(lambda xarr: np.abs(xarr))

# %% Load segments
dtc = xr.open_dataset("../data/dtc.nc")
segments = pd.read_csv(
    "../data/segments.csv", parse_dates=["starttime", "cpatime", "endtime"]
)

# %% Plot
plt.style.use("../figures.mplstyle")
blank = 0.2
bar = 0.07
fig, axes = plt.subplots(
    nrows=11,
    figsize=(7.1, 5),
    gridspec_kw=dict(
        hspace=0.0,
        wspace=0.0,
        left=0.06,
        right=0.98,
        bottom=0.05,
        top=0.99,
        height_ratios=[1, bar, 1, blank, 1, bar, 1, blank, 1, bar, 1],
    ),
)
for i in range(N):
    ax = axes[4 * i]
    ax.scatter(t, a, marker="s", s=4, fc="none", ec="C0", linewidth=0.5)
    for atrack in atracks:
        ax.plot(t, atrack, c="black", ls="-.")
    for idx, segment in segments.iterrows():
        ax.axvline(segment["starttime"], c="C2", ls=":")
        ax.axvline(segment["cpatime"], c="black", ls=":")
        ax.axvline(segment["endtime"], c="C3", ls=":")
    ax.set_xlim(date_range[i], date_range[i + 1])
    ax.tick_params(bottom=False, labelbottom=False)
    ax.set_ylim(0, 360)
    ax.set_yticks([0, 90, 180, 270, 360])
    ax.set_ylabel("Azimuth [°]")

    ax = axes[4 * i + 1]
    ax.fill_between(
        dtc["xor"]["time"],
        0,
        1 * dtc["xor"],
        facecolor="C0",
        lw=0,
        edgecolor="none",
        step="mid",
    )
    ax.fill_between(
        dtc["vm"]["time"],
        0,
        1 * dtc["vm"],
        facecolor="C2",
        lw=0,
        edgecolor="none",
        step="mid",
    )
    ax.fill_between(
        dtc["vp"]["time"],
        0,
        1 * dtc["vp"],
        facecolor="C3",
        lw=0,
        edgecolor="none",
        step="mid",
    )
    ax.tick_params(bottom=False, labelbottom=False, labelleft=False, left=False)
    ax.set_xlim(date_range[i], date_range[i + 1])
    for idx, segment in segments.iterrows():
        ax.axvline(segment["starttime"], c="C2", ls=":")
        ax.axvline(segment["cpatime"], c="black", ls=":")
        ax.axvline(segment["endtime"], c="C3", ls=":")

    ax = axes[4 * i + 2]
    ax.scatter(
        t,
        r / 1000,
        marker="$\u25A1$",
        s=4,
        c=np.clip(v.values * 1.943844, -25, 25),
        linewidth=0.5,
        cmap="cet_diverging_gwr_55_95_c38",
    )
    for rtrack in rtracks:
        ax.plot(t, rtrack / 1000, c="black", ls="-.")
    for idx, segment in segments.iterrows():
        ax.axvline(segment["starttime"], c="C2", ls=":")
        ax.axvline(segment["cpatime"], c="black", ls=":")
        ax.axvline(segment["endtime"], c="C3", ls=":")
    ax.annotate(
        date_range[i].strftime("%d/%m"), (3, 3), xycoords="axes points", color="black"
    )
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
    ["00:00", "06:00", "12:00", "18:00", "00:00", "06:00", "12:00", "18:00", "00:00"]
)

fig.align_labels()
fig.savefig("../figs/detection.pdf")
