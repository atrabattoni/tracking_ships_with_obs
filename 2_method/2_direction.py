# %% Import Libraries
import pickle

from colorcet import cm as cc
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
from obspy import read, read_inventory

import obsea


# %% Parameters

nperseg = 1024
dt = 180.0
t_step = 60
R = 0.99
fmin, fmax = 11.0, 24.0
water_level = 40.0
obs_orientation = 77.0
obs_location = 350.0 + 1j*50.0
cpa = 15_000.0
radius = 70_000.0
n_ship = 1
n_azimuth = 361
endpoint = True


# %% Load data

# Seimological
st = read("../data/waveform.mseed")
inventory = read_inventory("inventory.xml")
st.attach_response(inventory)

# AIS
with open("../data/track.pkl", "rb") as file:
    track = pickle.load(file)
track -= obs_location
track *= np.exp(1j*np.deg2rad(obs_orientation))

# Prepocessing
tf = obsea.time_frequency(
    st, nperseg=nperseg, step=nperseg//2, water_level=water_level)
s = obsea.spectrogram(tf["p"])
az = obsea.intensity(tf)
u = az.sel(frequency=slice(fmin, fmax))


# %% Tonal detection
ell = obsea.tonal_detection(
    u, n_azimuth, 0.0, R, dt, endpoint=endpoint, t_step=t_step)
track = track.interp_like(ell)

seuil = np.log(ell.mean("azimuth"))
peaks = ell.argmax("azimuth").astype(float)
peaks[seuil <= 0] = np.nan
peaks *= ell["azimuth"][1]

ell["time"] = pd.to_datetime(ell["time"].values, unit="s")
peaks["time"] = pd.to_datetime(peaks["time"].values, unit="s")
track["time"] = pd.to_datetime(track["time"].values, unit="s")
az["time"] = pd.to_datetime(az["time"].values, unit="s")


# %% Plot
plt.style.use("../figures.mplstyle")

fig, axes = plt.subplots(2, sharex=True, gridspec_kw=dict(
    hspace=0.08, wspace=0.0,
    left=0.14, right=0.97,
    bottom=0.08, top=0.98,
))
# Azigram
ax = axes[0]
img = obsea.plot_azigram(az, ax=ax, rasterized=True, add_colorbar=False)
fig.colorbar(img, ax=ax, pad=0.01, ticks=np.arange(0, 361, 60),
             label="Azimuth [°]")
ax.axhline(11, color="black", ls="--")
ax.axhline(24, color="black", ls="--")
ax.set_xlabel(None)
ax.set_yticks([0, 5, 10, 15, 20, 25])
ax.set_ylabel("Frequency [Hz]")

# Log-Likelihood
ax = axes[1]
img = ax.pcolormesh(ell["time"], ell["azimuth"], np.log(ell.T),
                    vmin=-40, vmax=40, cmap=cc.coolwarm, rasterized=True)
fig.colorbar(img, ax=ax, pad=0.01, label="Log-likelihood")
ax.plot(peaks["time"], peaks,
        ls="", marker="s", ms=2, mfc="none", mec="C2", label="detection")
ax.plot(track["time"], np.rad2deg(
    np.arctan2(track.real, track.imag)) % 360, "black", ls="-.", label="AIS")
ax.set_ylim(0, 360)
ax.yaxis.set_major_locator(MultipleLocator(60))
ax.yaxis.set_minor_locator(MultipleLocator(10))
ax.set_ylabel("Azimuth [°]")
# ax.legend(loc="lower right")

ax.set_xlim(
    np.datetime64("2012-11-27T06:30:00"),
    np.datetime64("2012-11-27T10:00:00"),
)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
fig.savefig("../figs/method_direction.pdf")

# %%
