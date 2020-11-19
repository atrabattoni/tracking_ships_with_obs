import pickle

import obsea
import numpy as np
from obspy import read_inventory

# Load station
station = read_inventory("../data/RR03.xml")[0][0]
timedelta = np.timedelta64(24, 'h')
radius = 100_000.0

# Method
dataset = 'ais_marine_traffic'
cpa = 15_000.0
n_ship = 1

fname = obsea.get_dataset_path(dataset)
ais = obsea.read_marine_traffic(fname)
tracks = obsea.read_ais(ais, timedelta)
tracks = obsea.select_tracks(tracks, station, radius, cpa)

track = tracks.iloc[n_ship]
with open("../data/track.pkl", "wb") as file:
    pickle.dump(track, file)

# Week
dataset = 'ais_week_cls'
cpa = 25_000.0
timedelta = np.timedelta64(24, 'h')

fname = obsea.get_dataset_path(dataset)
ais = obsea.read_cls(fname)
tracks = obsea.read_ais(ais, timedelta)
tracks = obsea.select_tracks(tracks, station, radius, cpa)

tracks.to_pickle("../data/tracks.pkl")
