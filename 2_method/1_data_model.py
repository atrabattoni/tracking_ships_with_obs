from scipy.stats import skewnorm
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# Parameters
N = 3
sigma = 0.05

# Load data
mu = xr.open_dataarray("../data/mu.nc")
ray_tracing = xr.open_dataset("../inputs/ray_tracing.nc")
ray_tracing = ray_tracing.sel(path=slice(1, N + 1))

# Compute TDOA/PDOA
toa = ray_tracing["toa"]
poa = np.arctan2(ray_tracing["amplitude_imag"], ray_tracing["amplitude_real"])
tdoa = toa.diff("path", label="lower").rename({"path": "interference"})
pdoa = poa.diff("path", label="lower").rename({"path": "interference"})


# Utils


def make_mask(sigma, tdoa, q):
    return np.exp(-(((q - tdoa) / sigma) ** 2))


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
    [amplitude_model(mu["distance"], models[i]) for i in [1, 2, 3]], axis=-1
)
amplitude = xr.DataArray(
    data=data,
    coords={"distance": mu["distance"], "interference": [1, 2, 3]},
    dims=("distance", "interference"),
)
mask = make_mask(sigma, tdoa, mu["quefrency"])
output = mask * amplitude
output = output.sum("interference")

# Complete model
f0 = 16.5
dtau = 0.05
tau = np.linspace(0, 10.24, 513)
tau = xr.DataArray(data=tau, coords={"quefrency": tau}, dims=["quefrency"])
mu_model = (
    amplitude
    * np.cos(2 * np.pi * (tau - tdoa) * f0 - pdoa)
    * np.exp(-((tau - tdoa) ** 2) / dtau**2)
)
sigma_model = (0.625 + np.cos(np.pi * tau / tau.max()) / 5) / np.sqrt(1024)
sigma_model = xr.DataArray(sigma_model, coords={"quefrency": tau}, dims=["quefrency"])

tdoa.to_netcdf("../data/tdoa_model.nc")
mu_model.to_netcdf("../data/mu_model.nc")
sigma_model.to_netcdf("../data/mu_sigma.nc")

# Limit to 50 km
tdoa.loc[{"distance": slice(50000, None)}] = np.nan
output.loc[{"distance": slice(50000, None)}] = 0

# %% Plot
plt.style.use("../figures.mplstyle")
fig, axes = plt.subplots(
    nrows=2,
    sharex=True,
    sharey=True,
    figsize=(3.4, 3.4),
    gridspec_kw=dict(
        hspace=0.06,
        wspace=0.0,
        left=0.09,
        right=1.03,
        bottom=0.10,
        top=0.98,
    ),
)

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
img = ax.pcolormesh(
    mu["distance"] / 1000, mu["quefrency"], output, vmin=0, vmax=0.1, rasterized=True
)
fig.colorbar(img, ax=ax, pad=0.02)
ax.set_xlabel("Distance [km]", labelpad=2)
ax.set_ylabel("Quefrency [s]", labelpad=2)

ax.set_xlim(0, 60)
ax.set_ylim(0, 7)
ax.set_yticks(np.arange(8))

fig.savefig("../figs/data_model.pdf")
