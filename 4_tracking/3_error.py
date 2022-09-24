import numpy as np
import pandas as pd

tracks = pd.read_pickle("../inputs/tracks.pkl")
lines = pd.read_pickle("../data/lines.pkl")

tracks = tracks.reset_index()
tracks.index += 1

lines["cpa_distance"] = np.abs(lines["cpa_distance"])
lines["speed_heading"] = (np.rad2deg(lines["speed_heading"]) + 77) % 360

# Ship Information

type2name = {
    70: "Cargo",
    71: "Cargo",
    79: "Cargo",
    80: "Tanker",
}

shipInfo = tracks[["mmsi", "name", "type", "length", "width", "draught"]]
shipInfo.columns = shipInfo.columns.str.capitalize()
shipInfo["Name"] = shipInfo["Name"].str.title()
shipInfo["Type"] = shipInfo["Type"].apply(lambda x: type2name[x])
shipInfo.to_excel("../data/shipInfo.xlsx")

error = tracks[["cpa_time", "cpa_distance", "speed_value", "speed_heading"]]
error["cpa_time"] = error["cpa_time"].astype(int) / 1e9
error = np.abs(lines - error)
error["speed_heading"] = error["speed_heading"] % 360
error["speed_value"] = error["speed_value"] * 1.94384

df = pd.DataFrame(
    {
        "cpa_distance": [
            np.round(error["cpa_distance"].iloc[:10].median() * 1.4826, 0)
        ],
        "cpa_time": [np.round(error["cpa_time"].median() * 1.4826, 0)],
        "speed_heading": [np.round(error["speed_heading"].median() * 1.4826, 1)],
        "speed_value": [np.round(error["speed_value"].iloc[:10].median() * 1.4826, 2)],
    }
)
df.to_excel("../data/track_general_errors.xlsx")

error["cpa_distance"] = np.round(error["cpa_distance"], 0)
error["cpa_time"] = np.round(error["cpa_time"], 0)
error["speed_heading"] = np.round(error["speed_heading"], 1)
error["speed_value"] = np.round(error["speed_value"], 2)
error.to_excel("../data/track_errors.xlsx")
