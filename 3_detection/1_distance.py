
import numpy as np
import obsea
import pandas as pd
import xarray as xr
from obspy import read, read_inventory

reference = 300 + 1j*550.0
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
mu = xr.open_dataarray(
    "/Users/alister/Desktop/PhD/thesis/src/bayes/data/mu_model.nc")
sigma = xr.open_dataarray(
    "/Users/alister/Desktop/PhD/thesis/src/bayes/data/sigma_model.nc")
tdoa = xr.open_dataarray(
    "/Users/alister/Desktop/PhD/thesis/src/bayes/data/tdoa_november.nc").T
model = obsea.build_model(mu, sigma, tdoa, 0.05, 50)

# Load waveforms
st = read("../data/waveforms.mseed")
inventory = read_inventory("../data/inventory.xml")
st.attach_response(inventory)
st = st.select(channel="BDH")

# Process waveforms
p = obsea.time_frequency(st, nperseg, step)["p"]
ceps = obsea.cepstrogram(p)
ceps = obsea.batch_svd_filter(ceps, 3*3600, remove_mean=False)

t = pd.date_range("2013-05-21", "2013-05-27", freq="1 min")
t = (t - pd.Timestamp(0)) / pd.Timedelta(1, "s")
ell = obsea.cepstral_detection(
    ceps, model, grid["dr"], grid["rmax"], grid["dv"], grid["vmax"],
    nsigma, grid["dt"], t=t)
marginal = ell.mean(["distance", "speed"])
mask = marginal > 1.0

arg = ell.argmax(["distance", "speed"])
r = grid["dr"] * arg["distance"]
v = grid["dv"] * arg["speed"] - grid["vmax"]
r[~mask] = np.nan
v[~mask] = np.nan

ell_r = ell.mean("speed")
ell_rm = ell.sel(speed=slice(None, 0)).mean("speed")
ell_rp = ell.sel(speed=slice(0, None)).mean("speed")

r.to_netcdf("../data/r.nc")
v.to_netcdf("../data/v.nc")
ell.to_netcdf("../data/ell_rv.nc")
ell_r.to_netcdf("../data/ell_r.nc")
ell_rm.to_netcdf("../data/ell_rm.nc")
ell_rp.to_netcdf("../data/ell_rp.nc")

