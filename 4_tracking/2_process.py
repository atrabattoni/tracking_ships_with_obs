import pickle

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import obsea
import pandas as pd
import xarray as xr
from dask import compute, delayed
from dask.diagnostics import ProgressBar
from obspy import read_inventory
from obspy.clients.fdsn import Client
from scipy.optimize import minimize

import utils


def process(segment, ntrack, data):

    # Parameters
    Pd = xr.Dataset({"a": 0.9, "r": 0.6})

    # Segment slicing
    t_cpa = utils.to_posix(segment[1])
    ell = utils.select_segment(data, segment, convert="posix")
    ell = utils.detection_probability(ell, Pd)

    # Line Fitting
    loglik = np.log(utils.smooth(ell, 14, 10, 1))  # 14 km, 5°, 0.5 m/s
    brute_force = utils.make_brute_force(loglik)

    # Coarse
    tg = t_cpa + np.arange(-30*60, 30*60 + 3*60, 3*60)
    rg = np.arange(-50_000, 50_000 + 1000, 1000)
    ag = np.arange(0, 2 * np.pi, np.deg2rad(10))
    vg = np.arange(6, 13 + 0.5, 0.5)
    index, _ = brute_force(tg, rg, ag, vg)
    t_cpa = tg[index[0]]
    r_cpa = rg[index[1]]
    a_inf = ag[index[2]]
    v_inf = vg[index[3]]

    # Medium
    tg = t_cpa + np.arange(-3*60, 3*60 + 10, 10)
    rg = r_cpa + np.arange(-1000, 1000 + 100, 100)
    ag = (a_inf + np.arange(-np.deg2rad(10),
                            np.deg2rad(10 + 1),
                            np.deg2rad(1))) % (2*np.pi)
    vg = v_inf + np.arange(-0.5, 0.5 + 0.1, 0.1)
    index, _ = brute_force(tg, rg, ag, vg)
    t_cpa = tg[index[0]]
    r_cpa = rg[index[1]]
    a_inf = ag[index[2]]
    v_inf = vg[index[3]]

    # Fine
    tg = t_cpa + np.arange(-10, 10 + 1, 1)
    rg = r_cpa + np.arange(-100, 100 + 10, 10)
    ag = (a_inf + np.arange(-np.deg2rad(1),
                            np.deg2rad(1 + 0.1),
                            np.deg2rad(0.1))) % (2*np.pi)
    vg = v_inf + np.arange(-0.1, 0.1 + 0.01, 0.01)
    index, _ = brute_force(tg, rg, ag, vg)
    t_cpa = tg[index[0]]
    r_cpa = rg[index[1]]
    a_inf = ag[index[2]]
    v_inf = vg[index[3]]

    return {
        "ntrack": ntrack,
        "cpa_time": t_cpa,
        "cpa_distance": r_cpa,
        "speed_heading": a_inf,
        "speed_value": v_inf,
    }


# Load likelihood
data = xr.Dataset({
    "a": xr.open_dataarray("ell_a.nc"),
    "r": xr.open_dataarray("ell_rv.nc"),
})

# Load segments
with open("segments.pkl", "rb") as file:
    segments = pickle.load(file)

ntracks = [1, 2, 3, 4, 5, 6, 8, 10, 11]
results = []
for ntrack, segment in zip(ntracks, segments):
    results.append(delayed(process)(segment, ntrack, data))

with ProgressBar():
    results = compute(results)

result = pd.concat([pd.DataFrame(res) for res in results])
result = result.set_index("ntrack")
result.to_pickle("lines.pkl")