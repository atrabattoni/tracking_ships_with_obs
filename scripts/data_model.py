from scipy.stats import skewnorm
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

mu = xr.open_dataarray(
    "/Users/alister/Desktop/PhD/thesis/src/warp/mu.nc")
r = np.loadtxt("/Users/alister/Desktop/PhD/thesis/src/ray-tracing/data/r.csv")
tau = np.loadtxt(
    "/Users/alister/Desktop/PhD/thesis/src/ray-tracing/data/tau.csv", delimiter=",")

dtau_1 = tau[:, 1:] - tau[:, :-1]
dtau_2 = tau[:, 2:] - tau[:, :-2]
dtau_3 = tau[:, 3:] - tau[:, :-3]

# %% Parameters
N = 3
sigma = 0.05

# %% Load TDOA
toa = xr.DataArray(
    data=tau[:, :N+1],
    coords={
        "distance": r,
        "interference": np.arange(N+1)
    },
    dims=["distance", "interference"],
)
tdoa = toa.diff("interference")

# %% Load amplitude data


def make_mask(sigma, tdoa, q):
    return np.exp(-((q - tdoa) / sigma)**2)


def amplitude_model(x, model):
    k, a, loc, scale = model
    return k * skewnorm.pdf(x, a, loc, scale)


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


plt.style.use("figures.mplstyle")
fig, axes = plt.subplots(
    nrows=2, sharex=True, sharey=True, figsize=(3.4, 3.4), gridspec_kw=dict(
        hspace=0.05, wspace=0.0,
        left=0.10, right=1.02,
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

# ax.plot(r / 1000, dtau_1, color="C1", alpha=0.4, ls="--")
# ax.plot(r / 1000, dtau_2, color="C3", alpha=0.4, ls="--")
# ax.plot(r / 1000, dtau_3, color="C6", alpha=0.4, ls="--")
# ax.plot(np.nan, np.nan, color="C1", label="2")
# ax.plot(np.nan, np.nan, color="C3", label="4")
# ax.plot(np.nan, np.nan, color="C6", label="6")
# ax.legend(title="Path difference", loc="upper left")

ax.set_xlim(0, 70)
ax.set_ylim(0, 10.24)
ax.set_ylabel("Quefrency [s]", labelpad=0)

ax = axes[1]
img = ax.pcolormesh(mu["distance"]/1000, mu["quefrency"],
                    output, vmin=0, vmax=0.1, rasterized=True)
fig.colorbar(img, ax=ax, pad=0.02)
ax.set_xlabel("Distance [km]", labelpad=2)
ax.set_ylabel("Quefrency [s]", labelpad=0)
ax.set_xlim(0, 70)

fig.savefig("figs/data_model_a.pdf")
