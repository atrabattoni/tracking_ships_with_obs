from glob import glob

import numpy as np
import pandas as pd
import obsea

fnames = sorted(glob("../data/track_*.nc"))
tracks = pd.Series([obsea.read_complex(fname) for fname in fnames])
measlines = pd.read_csv("../data/lines.csv", index_col="ntrack")

records = []
for idx, track in enumerate(tracks, start=1):
    data = {}
    cpa = obsea.get_cpa(track)
    time = cpa["time"].values[0]
    cpa = cpa.values[0]
    speed = obsea.simplify(track)["v"].values[0]
    data["ntrack"] = idx
    data["cpa_time"] = (time - np.datetime64(0, "s")) / np.timedelta64(1, "s")
    data["cpa_distance"] = np.abs(cpa)
    data["speed_heading"] = np.rad2deg(np.arctan2(speed.real, speed.imag)) % 360
    data["speed_value"] = np.abs(speed)
    records.append(data)
truelines = pd.DataFrame.from_records(records, index="ntrack")

measlines["cpa_distance"] = np.abs(measlines["cpa_distance"])
measlines["speed_heading"] = (np.rad2deg(measlines["speed_heading"]) + 77) % 360

error = np.abs(measlines - truelines)
error["speed_heading"] = error["speed_heading"] % 360
error["speed_value"] = error["speed_value"] * 1.94384

df = pd.DataFrame(
    {
        "cpa_time": [np.round(error["cpa_time"].median() * 1.4826, 0)],
        "cpa_distance": [
            np.round(error["cpa_distance"].iloc[:10].median() * 1.4826, 0)
        ],
        "speed_heading": [np.round(error["speed_heading"].median() * 1.4826, 1)],
        "speed_value": [np.round(error["speed_value"].iloc[:10].median() * 1.4826, 2)],
    }
)
df.to_csv("../data/track_general_errors.csv")

error["cpa_time"] = np.round(error["cpa_time"], 0)
error["cpa_distance"] = np.round(error["cpa_distance"], 0)
error["speed_heading"] = np.round(error["speed_heading"], 1)
error["speed_value"] = np.round(error["speed_value"], 2)
error.to_csv("../data/track_errors.csv")
