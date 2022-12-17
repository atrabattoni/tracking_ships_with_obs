"""
Compute radial log-likelihood.
"""


# %% Libs
import numpy as np
import obsea
import pandas as pd
import xarray as xr
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

# %% Load model
mu = xr.open_dataarray("../data/mu_model.nc").transpose(
    "interference", "distance", "quefrency"
)
sigma = xr.open_dataarray("../data/sigma_model.nc")
tdoa = xr.open_dataarray("../data/tdoa_model.nc").transpose("interference", "distance")
model = obsea.build_model(mu, sigma, tdoa, 0.05, 50)

# %% Load waveforms
st = read("../data/waveforms.mseed")
inventory = read_inventory("../data/RR03.xml")
st.attach_response(inventory)
st = st.select(channel="BDH")

# %% Process waveforms
p = obsea.time_frequency(st, nperseg, step)["p"]
ceps = obsea.cepstrogram(p)

t = pd.date_range("2013-05-21", "2013-05-27", freq="1 min")
t = (t - pd.Timestamp(0)) / pd.Timedelta(1, "s")


_ceps = ceps.copy()
_ceps["time"] = (_ceps["time"] - np.datetime64(0, "s")) / np.timedelta64(1, "s")
_ceps = obsea.batch_svd_filter(_ceps, 3 * 3600, remove_mean=False)
_ell = obsea.cepstral_detection(_ceps, model, grid, nsigma, t=t)
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

ell_r = ell.mean("speed")
ell_rm = ell.sel(speed=slice(None, 0)).mean("speed")
ell_rp = ell.sel(speed=slice(0, None)).mean("speed")

# %% Save
r.to_netcdf("../data/r.nc")
v.to_netcdf("../data/v.nc")
ell.to_netcdf("../data/ell_rv.nc")
ell_r.to_netcdf("../data/ell_r.nc")
ell_rm.to_netcdf("../data/ell_rm.nc")
ell_rp.to_netcdf("../data/ell_rp.nc")
