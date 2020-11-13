from scipy.stats import skewnorm
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# Parameters
N = 3
sigma = 0.05

# Load data
mu = xr.open_dataarray(
    "/Users/alister/Desktop/PhD/thesis/src/warp/mu.nc")
r = np.loadtxt("/Users/alister/Desktop/PhD/thesis/src/ray-tracing/data/r.csv")
tau = np.loadtxt(
    "/Users/alister/Desktop/PhD/thesis/src/ray-tracing/data/tau.csv", delimiter=",")

# Compute TDOA
toa = xr.DataArray(
    data=tau[:, :N+1],
    coords={
        "distance": r,
        "interference": np.arange(N+1)
    },
    dims=["distance", "interference"],
)
tdoa = toa.diff("interference")


# Utils 

def make_mask(sigma, tdoa, q):
    return np.exp(-((q - tdoa) / sigma)**2)


def amplitude_model(x, model):
    k, a, loc, scale = model
    return k * skewnorm.pdf(x, a, loc, scale)

# Amplitude model

models = {
    1: (8.9e2, 4.0, 8.0e3, 7.5e3),
    2: (1.6e3, 6.0, 1.4e4, 2.0e4),
    3: (1.0e3, 4.0, 2.7e4, 2.0e4),
}
data = np.stack(
    [amplitude_model(mu["distance"], models[i]) for i in [1, 2, 3]], axis=-1)
amplitude = xr.DataArray(
    data=data,
    coords={"distance": mu["distance"], "interference": [1, 2, 3]},
    dims=("distance", "interference")
)
mask = make_mask(sigma, tdoa, mu["quefrency"])
output = mask * amplitude
output = output.sum("interference")

# Limit to 50 km
tdoa.loc[{"distance": slice(50000, None)}] = np.nan
output.loc[{"distance": slice(50000, None)}] = 0

# Plot
plt.style.use("figures.mplstyle")
fig, axes = plt.subplots(
    nrows=2, sharex=True, sharey=True, figsize=(3.4, 3.4), gridspec_kw=dict(
        hspace=0.06, wspace=0.0,
        left=0.09, right=1.03,
        bottom=0.10, top=0.98,
    ))

ax = axes[0]
img = ax.pcolormesh(
    mu["distance"] / 1000,
    mu["quefrency"],
    mu,
    rasterized=True,
    vmin=0,
    vmax=0.1,
)
fig.colorbar(img, ax=ax, pad=0.02)
ax.plot(tdoa["distance"] / 1000, tdoa, color="C3", alpha=0.5, ls="--")
ax.set_ylabel("Quefrency [s]", labelpad=2)

ax = axes[1]
img = ax.pcolormesh(mu["distance"]/1000, mu["quefrency"],
                    output, vmin=0, vmax=0.1, rasterized=True)
fig.colorbar(img, ax=ax, pad=0.02)
ax.set_xlabel("Distance [km]", labelpad=2)
ax.set_ylabel("Quefrency [s]", labelpad=2)

ax.set_xlim(0, 60)
ax.set_ylim(0, 7)
ax.set_yticks(np.arange(8))

fig.savefig("figs/data_model.pdf")
