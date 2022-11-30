"""
Plot figure 6.
"""

# %% Libraries
import colorcet
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import obsea
import pandas as pd
import xarray as xr
from matplotlib.offsetbox import AnchoredText
from obspy import read, read_inventory

# %% Parameters
reference = 300 + 1j * 550.0
nperseg = 1024
step = 512
nsigma = 2
grid = {
    "dt": 180.0,
    "dr": 70.0,
    "rmax": 50_000.0,
    "dv": 0.5,
    "vmax": 13.0,
}

# %% Load models
mu = xr.open_dataarray("../data/mu_model.nc").transpose(
    "interference", "distance", "quefrency"
)
sigma = xr.open_dataarray("../data/sigma_model.nc")
tdoa = xr.open_dataarray("../data/tdoa_model.nc").transpose("interference", "distance")

model = obsea.build_model(mu, sigma, tdoa, 0.05, 50)

# %% Load waveforms
st = read("../data/waveform.mseed")
inventory = read_inventory("../data/RR03.xml")
st.attach_response(inventory)
st = st.select(channel="BDH")

# %% Process waveforms
p = obsea.time_frequency(st, nperseg, step)["p"]
ceps = obsea.cepstrogram(p)
ceps = obsea.svd_filter(ceps, remove_mean=False)

_ceps = ceps.copy()
_ceps["time"] = (_ceps["time"] - np.datetime64(0, "s")) / np.timedelta64(1, "s")
_ell = obsea.cepstral_detection(
    _ceps,
    model,
    grid["dr"],
    grid["rmax"],
    grid["dv"],
    grid["vmax"],
    nsigma,
    grid["dt"],
    t_step=60,
)
ell = _ell.copy()
ell["time"] = pd.to_datetime(ell["time"].values, unit="s")
marginal = (ell * ell["distance"]).sum("distance") / ell["distance"].sum("distance")
marginal = marginal.mean(["speed"])
mask = marginal > 1.0

arg = ell.argmax(["distance", "speed"])
r = grid["dr"] * arg["distance"]
v = grid["dv"] * arg["speed"] - grid["vmax"]
r[~mask] = np.nan
v[~mask] = np.nan

ceps = np.abs(obsea.analytic_signal(ceps))
loglik = np.log(ell.mean("speed"))

# %% Load AIS
track = obsea.read_complex("../data/track.nc")
track -= reference
track = track.interp_like(ceps)
rtrack = np.abs(track)

# %% Plot
plt.style.use("../figures.mplstyle")
fig, axes = plt.subplots(
    3,
    1,
    sharex=True,
    figsize=(3.4, 3.4),
    gridspec_kw=dict(
        hspace=0.1,
        wspace=0.0,
        left=0.12,
        right=1.0,
        bottom=0.11,
        top=0.98,
    ),
)

# Cepstra
ax = axes[0]
img = ax.pcolormesh(
    ceps["time"], ceps["quefrency"], ceps, vmin=0, vmax=0.1, rasterized=True
)
cbar = fig.colorbar(img, ax=ax, ticks=[0.0, 0.1], pad=0.02)
cbar.ax.text(
    1.19,
    0.5,
    "Value",
    rotation=90,
    verticalalignment="center",
    horizontalalignment="right",
    transform=ax.transAxes,
)
ax.set_ylabel("Quefrency [s]")
at = AnchoredText(
    "a)",
    prop=dict(size=10, weight="bold", color="white"),
    loc="upper left",
    frameon=False,
    borderpad=0,
    pad=0.2,
)
ax.add_artist(at)

# Log-Likelihood
ax = axes[1]
img = ax.pcolormesh(
    loglik["time"],
    loglik["distance"] / 1000,
    loglik.T,
    vmin=-200,
    vmax=200,
    cmap="cet_diverging_bwr_40_95_c42",
    rasterized=True,
)
cbar = fig.colorbar(img, ax=ax, pad=0.02, ticks=[-200, 0, 200])
cbar.ax.text(
    1.19,
    0.5,
    "Log-likelihood",
    rotation=90,
    verticalalignment="center",
    horizontalalignment="right",
    transform=ax.transAxes,
)
ax.set_ylim(0, 50)
ax.set_yticks([0, 25, 50])
ax.set_ylabel("Distance [km]")
at = AnchoredText(
    "b)",
    prop=dict(size=10, weight="bold"),
    loc="upper left",
    frameon=False,
    borderpad=0,
    pad=0.2,
)
ax.add_artist(at)

# Peaks
ax = axes[2]
sc = ax.scatter(
    r["time"],
    r.values / 1000,
    marker="$\u25A1$",
    s=4,
    c=np.clip(v.values * 1.943844, -25, 25),
    linewidths=0.5,
    label="detection",
    cmap="cet_diverging_gwr_55_95_c38",
)
cbar = fig.colorbar(sc, ax=ax, pad=0.02, ticks=[-25, 0, 25])
cbar.ax.text(
    1.19,
    0.5,
    "Speed [knots]",
    rotation=90,
    verticalalignment="center",
    horizontalalignment="right",
    transform=ax.transAxes,
)
ax.plot(rtrack["time"], rtrack / 1000, "black", ls="-.", label="AIS")
ax.set_ylim(0, 50)
ax.set_yticks([0, 25, 50])

ax.set_ylabel("Distance [km]")
ax.set_xlim(
    np.datetime64("2012-11-27T06:30:00"),
    np.datetime64("2012-11-27T10:00:00"),
)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.set_xlabel("Time")
at = AnchoredText(
    "c)",
    prop=dict(size=10, weight="bold"),
    loc="upper left",
    frameon=False,
    borderpad=0,
    pad=0.2,
)
ax.add_artist(at)

fig.savefig("../figs/method_distance.pdf")
