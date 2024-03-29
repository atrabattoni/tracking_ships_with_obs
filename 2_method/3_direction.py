"""
Plot figure 3.
"""

# %% Import Libraries
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import obsea
from colorcet import cm as cc
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import MultipleLocator
from obspy import read, read_inventory

# %% Parameters
nperseg = 1024
dt = 180.0
t_step = 60
R = 0.99
fmin, fmax = 11.0, 24.0
water_level = 40.0
obs_orientation = 77.0
obs_location = 350.0 + 1j * 50.0
cpa = 15_000.0
radius = 70_000.0
n_ship = 1
n_azimuth = 361
endpoint = True


# %% Load data

# Seimological
st = read("../data/waveform.mseed")
inventory = read_inventory("../data/RR03.xml")
st.attach_response(inventory)

# AIS
track = obsea.read_complex("../data/track.nc")
track -= obs_location
track *= np.exp(1j * np.deg2rad(obs_orientation))

# Prepocessing
tf = obsea.time_frequency(
    st, nperseg=nperseg, step=nperseg // 2, water_level=water_level
)
s = obsea.spectrogram(tf["p"])
az = obsea.intensity(tf)
u = az.sel(frequency=slice(fmin, fmax))


# %% Tonal detection
_u = u.copy()
_u["time"] = (_u["time"] - np.datetime64(1, "s")) / np.timedelta64(1, "s")
_ell = obsea.tonal_detection(
    _u, n_azimuth, 0.0, R, dt, endpoint=endpoint, t_step=t_step
)
ell = _ell.copy()
ell["time"] = (1e9 * ell["time"]).astype("datetime64[ns]")

track = track.interp_like(ell)

seuil = np.log(ell.mean("azimuth"))
peaks = ell.argmax("azimuth").astype(float)
peaks[seuil <= 0] = np.nan
peaks *= ell["azimuth"][1]

# %% Plot
plt.style.use("../figures.mplstyle")

fig, axes = plt.subplots(
    3,
    figsize=(3.4, 3.4),
    sharex=True,
    gridspec_kw=dict(
        hspace=0.1,
        wspace=0.0,
        left=0.14,
        right=0.97,
        bottom=0.11,
        top=0.98,
    ),
)
# Spectrogram
ax = axes[0]
img = s.plot(ax=ax, rasterized=True, add_colorbar=False, vmin=-80, vmax=-40)
fig.colorbar(img, ax=ax, pad=0.01, label="PSD [dB]")
ax.set_xlabel(None)
ax.set_yticks([0, 5, 10, 15, 20, 25])
ax.set_ylabel("Frequency [Hz]")
at = AnchoredText(
    "a)",
    prop=dict(size=10, weight="bold", color="white"),
    loc="upper left",
    frameon=False,
    borderpad=0,
    pad=0.2,
)
ax.add_artist(at)

# Azigram
ax = axes[1]
img = obsea.plot_azigram(az, ax=ax, rasterized=True, add_colorbar=False)
fig.colorbar(img, ax=ax, pad=0.01, ticks=np.arange(0, 361, 60), label="Azimuth [°]")
ax.axhline(11, color="black", ls="--")
ax.axhline(24, color="black", ls="--")
ax.set_xlabel(None)
ax.set_yticks([0, 5, 10, 15, 20, 25])
ax.set_ylabel("Frequency [Hz]")
at = AnchoredText(
    "b)",
    prop=dict(size=10, weight="bold"),
    loc="upper left",
    frameon=False,
    borderpad=0,
    pad=0.2,
)
ax.add_artist(at)

# Log-Likelihood
ax = axes[2]
img = ax.pcolormesh(
    ell["time"].values,
    ell["azimuth"].values,
    np.log(ell.T.values),
    vmin=-40,
    vmax=40,
    cmap=cc.coolwarm,
    rasterized=True,
)
fig.colorbar(img, ax=ax, pad=0.01, label="Log-likelihood")
ax.plot(
    peaks["time"],
    peaks,
    ls="",
    marker="s",
    ms=2,
    mfc="none",
    mec="C2",
    label="detection",
)
ax.plot(
    track["time"],
    np.rad2deg(np.arctan2(track.real, track.imag)) % 360,
    "black",
    ls="-.",
    label="AIS",
)
at = AnchoredText(
    "c)",
    prop=dict(size=10, weight="bold"),
    loc="upper left",
    frameon=False,
    borderpad=0,
    pad=0.2,
)
ax.add_artist(at)
ax.set_ylim(0, 360)
ax.yaxis.set_major_locator(MultipleLocator(60))
ax.yaxis.set_minor_locator(MultipleLocator(10))
ax.set_ylabel("Azimuth [°]")
ax.set_xlabel("Time")

ax.set_xlim(
    np.datetime64("2012-11-27T06:30:00"),
    np.datetime64("2012-11-27T10:00:00"),
)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
fig.savefig("../figs/method_direction.pdf")
