import numpy as np
import xarray as xr
import pandas as pd
import obsea
from obspy.clients.fdsn import Client
import matplotlib.pyplot as plt

# Inputs
client = Client('RESIF')
inventory = client.get_stations(network='YV', station='RR03', level='response')
network, = inventory
station, = network
ais_fname = obsea.get_dataset_path('ais_marine_traffic')
mmsi_list = pd.read_csv(
    obsea.get_dataset_path('mmsi_list'), squeeze=True).tolist()

# Paremeters
timedelta = pd.Timedelta(24, 'h')
radius = 100_000  # meters
cpa = 25_000  # meters
nperseg = 1024
step = 512
fs = 50.0
water_level = 40
obs = 350 + 1j*50
q = np.linspace(0.0, nperseg/fs/2, nperseg // 2 + 1)
res = 70.0

# Process AIS
ais = obsea.read_marine_traffic(ais_fname)
ais = obsea.select_ships(ais, mmsi_list)
global_tracks = obsea.read_ais(ais, timedelta)
tracks = obsea.select_tracks(
    global_tracks, station, radius, cpa)
Nk = len(tracks)


# Process data

def process(ship):

    track = tracks.iloc[ship]

    st = obsea.load_stream(
        track, client, inventory, station, 'BDH', nb_channels=1)
    p = obsea.time_frequency(st, nperseg, step, water_level)["p"]
    ceps = obsea.cepstrogram(p)
    ceps = obsea.svd_filter(ceps, remove_mean=False)
    # ceps = obsea.highpass(ceps, 'quefrency', freq=0.2)
    ceps = obsea.analytic_signal(ceps)
    ceps = np.abs(ceps)

    xtrack = obsea.track2xarr(track)
    xtrack = xtrack.interp_like(ceps)

    delayer = obsea.make_delay(xtrack)
    xp = np.linspace(-750, 750, 61)
    yp = np.linspace(0, 750, 31)
    beamform = obsea.make_beamform(xp, yp, ceps, delayer)
    image = np.abs(beamform(1502, 4340))
    i, j = np.unravel_index(np.argmax(image.values), image.shape)
    x, y = xp[j], yp[i]

    xtrack -= x + 1j*y

    index = np.argmin(np.abs(xtrack.values))

    # Ship leaving away
    xtrack_leaving = xtrack.isel(time=slice(index, None))
    ceps_leaving = ceps.isel(time=slice(index, None))
    C_leaving = xr.DataArray(
        data=ceps_leaving.values,
        coords={
            "quefrency": ceps_leaving["quefrency"].values,
            "distance": np.abs(xtrack_leaving).values
        },
        dims=["quefrency", "distance"]
    )

    rmin = res * (C_leaving["distance"].min().values // res + 1)
    rmax = res * (C_leaving["distance"].max().values // res - 1)
    rs = np.arange(rmin, rmax + res, res)
    C_leaving = C_leaving.interp(distance=rs)

    # Ship coming
    xtrack_coming = xtrack.isel(time=slice(None, index)).isel(
        time=slice(None, None, -1))
    ceps_coming = ceps.isel(time=slice(None, index)).isel(
        time=slice(None, None, -1))
    C_coming = xr.DataArray(
        data=ceps_coming.values,
        coords={
            "quefrency": ceps_coming["quefrency"].values,
            "distance": np.abs(xtrack_coming).values
        },
        dims=["quefrency", "distance"]
    )

    rmin = res * (C_coming["distance"].min().values // res + 1)
    rmax = res * (C_coming["distance"].max().values // res - 1)
    rs = np.arange(rmin, rmax + res, res)
    C_coming = C_coming.interp(distance=rs)

    return C_coming, C_leaving


# %% Process

C = {}
for k in [1, 3, 10, 16, 23, 30, 31, 32, 33, 35]:
    print(k)
    try:
        C[str(k)+"_coming"], C[str(k)+"_leaving"] = process(k)
    except:
        continue
ds = xr.Dataset(C)
ds.to_netcdf("data.nc")

std = (0.625 + np.cos(np.pi*q/q.max()) / 5) / np.sqrt(nperseg)
std = xr.DataArray(
    data=std,
    coords={"quefrency": q},
    dims=["quefrency"],
)
x = ds.to_array()
x = (x / std)**2
mean = x.mean("variable")
mu = np.sqrt(mean - 2) * std
mu.values = np.nan_to_num(mu.values)
mu.to_netcdf("mu.nc")