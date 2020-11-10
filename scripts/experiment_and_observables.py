import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
from obspy import read_inventory

import obsea

# AIS
s_ais = obsea.read_cls("/Volumes/SSD/data/AIS/cls.csv", cargo_and_tanker=False)
t_ais = obsea.read_marine_traffic("/Volumes/SSD/data/AIS/marine_traffic.csv")

# Stations
inventory = read_inventory('/Volumes/SSD/data/StationXML/RR.xml')
network, = inventory

# Geographic projection
projection = ccrs.PlateCarree()

# Build Frame
plt.style.use("figures.mplstyle")
fig = plt.figure()
ax = fig.add_subplot(projection=projection)

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

# Coastline
coastline = cfeature.NaturalEarthFeature(
    category='physical', name='coastline', scale='10m',
    ec='black', fc=cfeature.COLORS["land"], lw=0.5)
ax.add_feature(coastline)

# Stations
for station in network:
    ax.plot(
        station.longitude, station.latitude, '*',
        markerfacecolor='white', markeredgecolor='black',
        markersize=6)


# Formatting Ticks
@FuncFormatter
def eastfmt(x, pos):
    return '{:g}°E'.format(x)


@FuncFormatter
def southfmt(x, pos):
    return '{:g}°S'.format(abs(x))


ax.set_xlim([47, 71])
ax.set_ylim([-35, -16])
ax.set_xticks([])
ax.set_yticks([])
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.yaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_minor_locator(MultipleLocator(1))
ax.xaxis.set_major_formatter(eastfmt)
ax.yaxis.set_major_formatter(southfmt)
ax.tick_params(direction='in', which="both")

fig.savefig("ais.pdf")
