import datetime as dt

import numpy as np
import obsea
import pandas as pd
from obspy import read_inventory
from shapely.geometry import Point


def parser(x):
    return dt.datetime(
        int(x[6:10]),
        int(x[3:5]),
        int(x[0:2]),
        int(x[11:13]),
        int(x[14:16]),
        int(x[17:19]))


# Parameters
obs_location = 350.0 + 1j*50.0
cpa = 25_000.0
radius = 100_000.0

# Load OBS
station = read_inventory("../data/inventory.xml")[0][0]
obs = Point(obs_location.real, obs_location.imag)

# Load AIS
raw = pd.read_csv("/Volumes/SSD/data/AIS/cls.csv", sep=";")
s = raw.locDate + ' ' + raw.locTime
raw.loc[:, 'timestamp'] = s.apply(parser)
ais = obsea.read_cls("/Volumes/SSD/data/AIS/cls.csv")

# Select period
query = (
    (raw["timestamp"] > pd.datetime(2013, 5, 20))
    & (raw["timestamp"] < pd.datetime(2013, 5, 28))
)
raw = raw[query]
query = (
    (ais["timestamp"] > pd.datetime(2013, 5, 20))
    & (ais["timestamp"] < pd.datetime(2013, 5, 28))
)
ais = ais[query]

# Select area
global_tracks = obsea.read_ais(ais, timedelta=pd.Timedelta(24, 'h'))
tracks = obsea.select_tracks(
    global_tracks, station, radius=radius, cpa=cpa)
tracks = tracks.to_frame("linestring")


# Add features
tracks["cpa_time"] = pd.to_datetime(
    tracks["linestring"].apply(
        lambda line: line.interpolate(line.project(obs)).coords[0][2]
    ),
    unit="s",
)
tracks["cpa_distance"] = tracks["linestring"].apply(obs.distance)
tracks["speed_value"] = tracks["linestring"].apply(
    lambda line: line.length / (line.coords[-1][2] - line.coords[0][2])
)
tracks["speed_heading"] = tracks["linestring"].apply(
    lambda line: np.rad2deg(np.arctan2(
        line.coords[-1][0] - line.coords[0][0],
        line.coords[-1][1] - line.coords[0][1],
    )) % 360
)
tracks["name"] = tracks.index.to_series().apply(
    lambda mmsi: raw[raw["mmsi"] == mmsi]["shipName"].unique()[0]
)
tracks["length"] = tracks.index.to_series().apply(
    lambda mmsi: raw[raw["mmsi"] == mmsi]["shipLength"].unique()[0]
)
tracks["width"] = tracks.index.to_series().apply(
    lambda mmsi: raw[raw["mmsi"] == mmsi]["shipWidth"].unique()[0]
)
tracks["draught"] = tracks.index.to_series().apply(
    lambda mmsi: np.median(raw[raw["mmsi"] == mmsi]["shipDraught"].unique())
)
tracks["type"] = tracks.index.to_series().apply(
    lambda mmsi: raw[raw["mmsi"] == mmsi]["aisShipType"].unique()[0]
)

# Filter and Sort tracks
tracks = tracks[tracks["cpa_time"] > pd.Timestamp("2013-05-21")]
tracks = tracks[tracks["cpa_time"] < pd.Timestamp("2013-05-27")]
tracks = tracks.sort_values("cpa_time")
tracks = tracks.reset_index()
tracks.index += 1

# Save
tracks.to_pickle("..data/tracks.pkl")
