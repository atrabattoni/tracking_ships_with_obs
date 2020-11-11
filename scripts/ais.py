import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.ticker import FuncFormatter, MultipleLocator
from obspy import read_inventory

import obsea

# AIS
s_ais = obsea.read_cls("/Volumes/SSD/data/AIS/cls.csv", cargo_and_tanker=False)
t_ais = obsea.read_marine_traffic("/Volumes/SSD/data/AIS/marine_traffic.csv")

# Stations
inventory = read_inventory("/Volumes/SSD/data/StationXML/RR.xml")
network, = inventory
rr03, = network.select(station="RR03")

# Geographic projection
projection = ccrs.PlateCarree()

# Build Frame
plt.style.use("figures.mplstyle")
fig, ax = plt.subplots(
    subplot_kw=dict(projection=projection),
    gridspec_kw=dict(
        left=0.05, right=1.0,
        bottom=0.06, top=0.99,
    )
)

# AIS logs
coeff = 2e-3
ax.scatter(
    s_ais["lon"], s_ais["lat"],
    s=coeff, ec="none", fc="C2",
    rasterized=True)
ax.scatter(
    t_ais["lon"], t_ais["lat"],
    s=12*coeff, ec="none", fc="C0",
    rasterized=True)
ax.scatter([], [], ec="none", fc="C2", label="S-AIS")
ax.scatter([], [], ec="none", fc="C0", label="T-AIS")

# Coastline
coastline = cfeature.NaturalEarthFeature(
    category="physical", name="coastline", scale="10m",
    ec="black", fc=cfeature.COLORS["land"], lw=0.5)
ax.add_feature(coastline)

# Stations
for station in network:
    ax.plot(
        station.longitude, station.latitude, "*",
        mfc="white", mec="black", ms=6)
ax.plot(
    rr03.longitude, rr03.latitude, "*",
    mfc="white", mec="C3", ms=6, mew=1.0)
ax.annotate(
    rr03.code, (rr03.longitude, rr03.latitude),
    xytext=(0, -9), textcoords="offset points",
    ha="center", c="white", weight="bold",
    path_effects=[
        path_effects.Stroke(linewidth=2, foreground='C3'),
        path_effects.Normal()])


# Formatting Ticks

@FuncFormatter
def eastfmt(x, pos):
    return "{:g}°E".format(x)


@FuncFormatter
def southfmt(x, pos):
    return "{:g}°S".format(abs(x))


ax.set_xlim([47, 71])
ax.set_ylim([-35, -16])
ax.set_xticks([])
ax.set_yticks([])

ax.legend(loc="lower right")

ax.xaxis.set_major_locator(MultipleLocator(5))
ax.yaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_minor_locator(MultipleLocator(1))
ax.xaxis.set_major_formatter(eastfmt)
ax.yaxis.set_major_formatter(southfmt)
ax.tick_params(direction="in", which="both")

fig.savefig("figs/ais.pdf")
