import numpy as np
from obspy import read, read_inventory
import obsea
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Dates
date_range = np.arange("2013-05-21", "2013-05-28", 2, dtype="datetime64[D]")
N = len(date_range) - 1

# Load
st = read("../data/waveforms.mseed")
inventory = read_inventory("../data/RR03.xml")
st.attach_response(inventory)
st = st.select(channel="BDH")

# Process
X = obsea.time_frequency(st, 1024, 512, water_level=40)
p = obsea.spectrogram(X["p"])

# Plot
plt.style.use("../figures.mplstyle")
fig, axes = plt.subplots(nrows=N, figsize=(7.1, 3), gridspec_kw=dict(
    hspace=0.11, wspace=0.0, left=0.06, right=0.97, bottom=0.07, top=0.98))

for i in range(N):
    ax = axes[i]
    img = ax.pcolormesh(p["time"], p["frequency"], p,
                        vmin=-100, vmax=0, cmap="viridis", rasterized=True)
    ax.annotate(date_range[i].item().strftime("%d/%m"),
                (3, 3), xycoords='axes points', color="white")
    ax.set_xlim(date_range[i], date_range[i + 1])
    ax.tick_params(labelbottom=False)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax.set_yticks([0, 5, 10, 15, 20, 25])

ax.tick_params(labelbottom=True)
ax.set_xticklabels(
    ["00:00", "06:00", "12:00", "18:00",
     "00:00", "06:00", "12:00", "18:00",
     "00:00"])
axes[N // 2].set_ylabel("Frequency [Hz]")

fig.savefig("../figs/spectrogram.pdf")
