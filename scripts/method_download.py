import pickle

import numpy as np
import obsea
import pandas as pd
from obspy import read_inventory
from obspy.clients.fdsn import Client

# Parameters
cpa = 15_000.0
radius = 70_000.0
n_ship = 1

# Load station
inventory = read_inventory("inventory.xml")
network, = inventory
station, = network

# Load AIS
ais = obsea.read_marine_traffic(obsea.get_dataset_path('ais_marine_traffic'))
mmsi_list = pd.read_csv(obsea.get_dataset_path(
    'mmsi_list'), squeeze=True).tolist()
ais = obsea.select_ships(ais, mmsi_list)

# Process AIS
global_tracks = obsea.read_ais(ais, timedelta=pd.Timedelta(24, 'h'))
local_tracks = obsea.select_tracks(
    global_tracks, station, radius=radius, cpa=cpa)
track = local_tracks.iloc[n_ship]

# Save track
with open("method_track.pkl", "wb") as file:
    pickle.dump(track, file)

# Download associated waveforms
client = Client("RESIF")
st = obsea.load_stream(track, client, inventory, station, '*')
st.write("method_waveforms.mseed")