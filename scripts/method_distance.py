import colorcet
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtickers
import numpy as np
import obsea
import pandas as pd
import xarray as xr
import pickle
from obspy import read, read_inventory

reference = 300 + 1j*550.0
nperseg = 1024
step = 512
nsigma = 2
grid = {
    "dt": 180,
    "dr": 70,
    "rmax": 50_000,
    "dv": 0.5,
    "vmax": 13,
}
mu = xr.open_dataarray(
    "/Users/alister/Desktop/PhD/thesis/src/bayes/data/mu_model.nc")
sigma = xr.open_dataarray(
    "/Users/alister/Desktop/PhD/thesis/src/bayes/data/sigma_model.nc")
tdoa = xr.open_dataarray(
    "/Users/alister/Desktop/PhD/thesis/src/bayes/data/tdoa_november.nc").T
ell0 = xr.open_dataarray(
    "/Users/alister/Desktop/PhD/thesis/src/bayes/data/ell.nc").T
model = obsea.build_model(mu, sigma, tdoa, 0.05, 50)

# Load waveforms
st = read("method_waveforms.mseed")
inventory = read_inventory("inventory.xml")
st.attach_response(inventory)
st = st.select(channel="BDH")

# Process waveforms
p = obsea.time_frequency(st, nperseg, step)["p"]
ceps = obsea.cepstrogram(p)
ceps = obsea.svd_filter(ceps, remove_mean=False)

data = obsea.compute_logell(ceps.values, **model)
logell = xr.DataArray(
    data=data,
    coords={
        "interference": [1, 2, 3],
        "distance": grid["dr"] * np.arange(data.shape[1]),  # TODO
        "time": ceps["time"],
    },
    dims=("interference", "distance", "time")
)

ell = obsea.cepstral_detection(
    ceps, model, grid["dr"], grid["rmax"], grid["dv"], grid["vmax"],
    nsigma, grid["dt"], t_step=60)
marginal = ell.mean(["distance", "speed"])
mask = marginal > 1.0

arg = ell.argmax(["distance", "speed"])
r = grid["dr"] * arg["distance"]
v = grid["dv"] * arg["speed"] - grid["vmax"]
r[~mask] = np.nan
v[~mask] = np.nan

ceps = np.abs(obsea.analytic_signal(ceps))
loglik = np.log(ell.mean("speed"))

# Load AIS
with open("method_track.pkl", "rb") as file:
    track = pickle.load(file)
xtrack = obsea.track2xarr(track)
xtrack -= reference
xtrack = xtrack.interp_like(ceps)
rtrack = np.abs(xtrack)

ceps["time"] = pd.to_datetime(ceps["time"].values, unit="s")
loglik["time"] = pd.to_datetime(loglik["time"].values, unit="s")
r["time"] = pd.to_datetime(r["time"].values, unit="s")
rtrack["time"] = pd.to_datetime(rtrack["time"].values, unit="s")


# %% PLOT

plt.style.use("figures.mplstyle")
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(3.4, 3.4), gridspec_kw=dict(
    hspace=0.1, wspace=0.0,
    left=0.12, right=1.0,
    bottom=0.06, top=0.98,
))
# Cepstra
ax = axes[0]
img = ax.pcolormesh(ceps["time"], ceps["quefrency"], ceps,
                    vmin=0, vmax=0.1, rasterized=True)
fig.colorbar(img, ax=ax, ticks=[0.0, 0.1], label="Value", pad=0.02)
ax.set_ylabel("Quefrency [s]")
# Log-Likelihood
ax = axes[1]
img = ax.pcolormesh(loglik["time"], loglik["distance"]/1000, loglik.T,
                    vmin=-200, vmax=200, cmap="cet_diverging_bwr_40_95_c42",
                    rasterized=True)
fig.colorbar(img, ax=ax, label="Log-likelihood", pad=0.02, ticks=[-200, 0, 200])
ax.set_ylim(0, 50)
ax.set_yticks([0, 25, 50])
ax.set_ylabel("Distance [km]")

# Peaks
ax = axes[2]
sc = ax.scatter(r["time"], r/1000, marker="s", s=4, fc="none",
                c=v*1.943844, linewidths=0.5, label="detection", cmap="cet_diverging_gwr_55_95_c38")
fig.colorbar(sc, ax=ax, label="Speed [knots]", pad=0.02,
             ticks=[-25, 0, 25])
ax.plot(rtrack["time"], rtrack/1000, "black", ls="-.", label="AIS")
ax.set_ylim(0, 50)
ax.set_yticks([0, 25, 50])

ax.set_ylabel("Distance [km]")
ax.set_xlim(
    np.datetime64("2012-11-27T06:30:00"),
    np.datetime64("2012-11-27T10:00:00"),
)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

fig.savefig("figs/method_distance.pdf")

# %%
