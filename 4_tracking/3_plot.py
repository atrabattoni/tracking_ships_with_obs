import pickle

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import numpy as np
import obsea
import pandas as pd
import xarray as xr
from colorcet import cm as cc

import utils

# Load segments
with open("segments.pkl", "rb") as file:
    segments = pickle.load(file)

# Load lines
lines = pd.read_pickle("lines.pkl")

# Load loglik
loglik_a = np.log(0.1 + 0.9 * xr.open_dataarray("../data/ell_a.nc"))
loglik_r = np.log(0.4 + 0.6 * xr.open_dataarray("../data/ell_r.nc"))

# Load tracks
tracks = pd.read_pickle("../data/tracks.pkl")
tracks_interp = tracks.apply(lambda xarr: xarr.interp_like(loglik_a["time"]))
atracks = tracks.apply(lambda xarr: (np.rad2deg(
    np.arctan2(xarr.real, xarr.imag)) - 77) % 360)
atracks_interp = tracks_interp.apply(lambda xarr: (np.rad2deg(
    np.arctan2(xarr.real, xarr.imag)) - 77) % 360)
rtracks = tracks.apply(lambda xarr: np.abs(xarr))
rtracks_interp = tracks_interp.apply(lambda xarr: np.abs(xarr))


def plot(fig, cell, nsegment, ntrack):
    segment = segments[nsegment]
    line = lines.loc[ntrack]
    track = tracks.loc[ntrack]
    track_interp = tracks_interp.loc[ntrack]
    atrack = atracks.loc[ntrack]
    atrack_interp = atracks_interp.loc[ntrack]
    rtrack = rtracks.loc[ntrack]
    rtrack_interp = rtracks_interp.loc[ntrack]
    t = pd.to_datetime(track["time"].values, unit="s")
    t_interp = pd.to_datetime(track_interp["time"].values, unit="s")
    t_ais = utils.select_segment(
        track, segment, convert="posix")["time"].values
    t_line = utils.select_segment(
        track_interp, segment, convert="posix")["time"].values
    la = utils.select_segment(loglik_a, segment, convert="posix")
    lr = utils.select_segment(loglik_r, segment, convert="posix")
   
    r, a, _ = utils.generate_line(line["cpa_time"], line["cpa_distance"],
                                  line["speed_heading"], line["speed_value"], t_line)
    x = r * np.sin(a + np.deg2rad(77))
    y = r * np.cos(a + np.deg2rad(77))

    r_ais, a_ais, _ = utils.generate_line(line["cpa_time"], line["cpa_distance"],
                                          line["speed_heading"], line["speed_value"], t_ais)
    x_ais = r_ais * np.sin(a_ais + np.deg2rad(77))
    y_ais = r_ais * np.cos(a_ais + np.deg2rad(77))

    inner_grid = cell.subgridspec(2, 2, wspace=0.05, hspace=0, width_ratios=[2.4, 1])

    # direction
    ax = fig.add_subplot(inner_grid[0, 0])
    ax.pcolormesh(pd.to_datetime(t_line, unit="s"),
                  la["azimuth"], la.T, rasterized=True, cmap=cc.coolwarm,
                  vmin=-40, vmax=40)
    ax.plot(t, atrack,
            marker="o", c="black", mfc="none", ls="", ms=2)
    ax.plot(t_interp, atrack_interp,
            c="black", ls="-.")
    ax.plot(pd.to_datetime(t_line, unit="s"), np.rad2deg(a) % 360, "C2")
    ax.plot(pd.to_datetime(t_ais, unit="s"), np.rad2deg(a_ais) % 360, "C2",
            ls="", marker="s", ms=2, mfc="none")
    ax.tick_params(labelbottom=False, bottom=False, pad=1)
    ax.set_ylim(0, 360)
    ax.set_xlim(segment[0], segment[2])
    ax.yaxis.set_major_locator(mticker.MultipleLocator(180))

    # distance
    ax = fig.add_subplot(inner_grid[1, 0])
    ax.pcolormesh(pd.to_datetime(t_line, unit="s"),
                  lr["distance"] / 1000, lr.T, rasterized=True, cmap=cc.coolwarm,
                  vmin=-100, vmax=100)
    ax.plot(t, rtrack / 1000,
            marker="o", c="black", mfc="none", ls="", ms=2)
    ax.plot(t_interp, rtrack_interp / 1000,
            c="black", ls="-.")
    ax.plot(pd.to_datetime(t_line, unit="s"), r / 1000, "C2")
    ax.plot(pd.to_datetime(t_ais, unit="s"), r_ais / 1000, "C2",
            ls="", marker="s", ms=2, mfc="none")
    ax.set_ylim(0, 50)
    ax.set_xlim(segment[0], segment[2])
    ax.yaxis.set_major_locator(mticker.MultipleLocator(25))

    locator = mdates.HourLocator(range(0, 24, 2))
    formatter = mdates.DateFormatter("%H:%M")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(pad=1)

    ax.annotate(ntrack, (3, 3), xycoords='axes points', color="C2")

    # map
    ax = fig.add_subplot(inner_grid[:, 1])
    circle50 = plt.Circle((0, 0), 50, ec="black", fc="none", lw=0.5, ls=":")
    circle100 = plt.Circle((0, 0), 100, ec="black", fc="none", lw=0.5, ls=":")
    ax.add_artist(circle50)
    ax.add_artist(circle100)
    ax.plot(0, 0, "*", mec="black", mfc="white", ms=5)
    ax.plot(track.real / 1000, track.imag / 1000,
            marker="o", c="black", mfc="none", ls="", ms=2)
    ax.plot(track_interp.real / 1000, track_interp.imag / 1000,
            c="black", ls="-.")
    ax.plot(x / 1000, y / 1000, "C2")
    ax.plot(x_ais / 1000, y_ais / 1000, "C2",
            ls="", marker="s", ms=2, mfc="none")
    ax.set_aspect("equal")
    ax.set_xlim([-100, 100])
    ax.set_ylim([-100, 100])
    ax.xaxis.set_major_locator(mticker.MultipleLocator(50))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(50))
    ax.tick_params(labelleft=False, pad=1)


ntracks = [1, 2, 3, 4, 5, 6, 8, 10, 11]
plt.style.use("../figures.mplstyle")
fig = plt.figure(figsize=(7.1, 5.5))
outer_grid = fig.add_gridspec(5, 2,
    hspace=0.25, wspace=0.15,
    left=0.04, right=0.99,
    bottom=0.03, top=0.99,
)
k = 0
for i in range(5):
    for j in range(2):
        try:
            nsegment = k
            ntrack = ntracks[k]
            cell = outer_grid[i, j]
            plot(fig, cell, nsegment, ntrack)
            k += 1
        except IndexError:
            pass
fig.savefig("../figs/tracking.pdf")
