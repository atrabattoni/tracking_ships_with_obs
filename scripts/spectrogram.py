import numpy as np
from obspy import read, read_inventory
import obsea
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Dates
date_range = pd.date_range("2013-05-21", "2013-05-27", freq="2D")
N = len(date_range) - 1

# Load
st = read("waveforms.mseed")
inventory = read_inventory("inventory.xml")
st.attach_response(inventory)
st = st.select(channel="BDH")

# Process
x = obsea.time_frequency(st, 1024, 512, water_level=40)
x = x["p"]
x = obsea.spectrogram(x)

x["time"] = pd.to_datetime(x["time"].values, unit="s")
chunks = [x.sel(time=slice(date_range[i], date_range[i + 1]))
          for i in range(N)]

# Plot
plt.style.use("figures.mplstyle")
fig, axes = plt.subplots(nrows=N, figsize=(7.1, 3), gridspec_kw={
    "hspace": 0.1, "wspace": 0.1,
})
for i in range(N):
    img = axes[i].pcolormesh(
        chunks[i]["time"], chunks[i]["frequency"], chunks[i],
        vmin=-100, vmax=0, cmap="viridis", rasterized=True
    )
    axes[i].annotate(date_range[i].strftime("%d/%m"), (3, 3),
                     xycoords='axes points', color="white")
    axes[i].set_xlim(date_range[i], date_range[i + 1])
    axes[i].tick_params(labelbottom=False)
    axes[i].xaxis.set_major_locator(mdates.HourLocator(interval=6))
    axes[i].set_yticks([0, 5, 10, 15, 20, 25])

axes[-1].tick_params(labelbottom=True)
axes[-1].set_xticklabels(
    ["00:00", "06:00", "12:00", "18:00",
     "00:00", "06:00", "12:00", "18:00",
     "00:00"])
axes[N // 2].set_ylabel("Frequency [Hz]")

fig.savefig("figs/spectrogram.pdf")
