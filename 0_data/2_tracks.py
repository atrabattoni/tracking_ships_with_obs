"""
Process the AIS data. 

AIS positions are converted into tracks. Cooridnates are projected onto a local 
cooridinate reference system centered onto the station location. Ship passing close 
enought are selected.
"""

# %% Libs
import numpy as np
import obsea
from obspy import read_inventory

# %% Load station
station = read_inventory("../data/RR03.xml")[0][0]
timedelta = np.timedelta64(24, "h")
radius = 100_000.0

# %% Method
dataset = "ais_marine_traffic"
cpa = 15_000.0
n_ship = 1

fname = obsea.get_dataset_path(dataset)
ais = obsea.read_marine_traffic(fname)
tracks = obsea.read_ais(ais, timedelta)
tracks = obsea.select_tracks(tracks, station, radius, cpa)

track = tracks.iloc[n_ship]
obsea.save_complex(track, "../data/track.nc")

# %% Week
dataset = "ais_week_cls"
cpa = 25_000.0
timedelta = np.timedelta64(24, "h")

fname = obsea.get_dataset_path(dataset)
ais = obsea.read_cls(fname)
tracks = obsea.read_ais(ais, timedelta)
tracks = obsea.select_tracks(tracks, station, radius, cpa)

# sort by cpa time
cpa_time = tracks.apply(lambda track: obsea.get_cpa(track)["time"].values[0])
mask = (np.datetime64("2013-05-21") <= cpa_time) & (
    cpa_time < np.datetime64("2013-05-28")
)
tracks = tracks[mask]
cpa_time = cpa_time[mask]
cpa_time = cpa_time.sort_values()
tracks = tracks[cpa_time.index]

for idx, track in enumerate(tracks, start=1):
    obsea.save_complex(track, f"../data/track_{idx:06d}.nc")
