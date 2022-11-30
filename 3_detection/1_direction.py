"""
Compute azimuthal log-likelihood.
"""

# %% Libs
import numpy as np
import pandas as pd
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
obs_location = 350.0 + 1j * 50.0
cpa = 15_000.0
radius = 70_000.0
n_ship = 1
n_azimuth = 361
endpoint = True

# $$ Load data
st = read("../data/waveforms.mseed")
inventory = read_inventory("../data/RR03.xml")
st.attach_response(inventory)

# %% Prepocessing
tf = obsea.time_frequency(
    st, nperseg=nperseg, step=nperseg // 2, water_level=water_level
)
s = obsea.spectrogram(tf["p"])
az = obsea.intensity(tf)
u = az.sel(frequency=slice(fmin, fmax))

# %% Tonal detection
t = pd.date_range("2013-05-21", "2013-05-27", freq="1 min")
t = (t - pd.Timestamp(0)) / pd.Timedelta(1, "s")
_u = u.copy()
_u["time"] = (_u["time"] - np.datetime64(1, "s")) / np.timedelta64(1, "s")
_ell = obsea.tonal_detection(_u, n_azimuth, 0.0, R, dt, endpoint=endpoint, t=t)
ell = _ell.copy()
ell["time"] = (1e9 * ell["time"]).astype("datetime64[ns]")

seuil = np.log(ell.mean("azimuth"))
peaks = ell.argmax("azimuth").astype(float)
peaks[seuil <= 0] = np.nan
peaks *= ell["azimuth"][1]

# %% Save
peaks.to_netcdf("../data/a.nc")
ell.to_netcdf("../data/ell_a.nc")
