# %%
import numpy as np
import xarray as xr
from scipy.stats import skewnorm
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

plt.style.use("../figures.mplstyle")

# %% TDOA Models
N = 3
tdoa = xr.open_dataarray("../data/tdoa_model.nc").sel(interference=slice(1, N))
r = tdoa["distance"]

f = [
    interp1d(r, tdoa[:, k], kind="linear", fill_value=np.nan, bounds_error=False)
    for k in range(N)
]

fs = [
    interp1d(tdoa[:, k], r, kind="linear", fill_value=np.nan, bounds_error=False)
    for k in range(N)
]


# %% Trajectory
v = 8.0  # m/s
d = 7000
t = 2 * 3600 * np.linspace(-1, 1, 1001)
r = np.abs(v * t + 1j * d)

tdoas = [f[k](r) for k in range(N)]

amp = []
a13 = (r > 7000) & (r < 18000)
a35 = (r > 12000) & (r < 34000)
a57 = (r > 23000) & (r < 40000)
amp = [a13, a35, a57]


# %% PLOT

fig, axes = plt.subplots(
    nrows=2,
    sharex=True,
    gridspec_kw=dict(
        hspace=0.08,
        wspace=0.0,
        left=0.12,
        right=0.96,
        bottom=0.08,
        top=0.98,
    ),
)
ax = axes[0]
for k in range(N):
    ax.scatter(x=t / 3600, y=tdoas[k], s=3 * amp[k], c="yellow", linewidths=0)
ax.set_xlim(-2, 2)
ax.set_ylim(0, 7)
ax.set_yticks(np.arange(8))
ax.set_ylabel("Quefrency [s]")
ax.set_facecolor("darkcyan")

ax = axes[1]
c = ["C0", "C3", "C2"]
for i in [1, 0, 2]:
    for j in range(N):
        x = t / 3600
        y = (fs[i](tdoas[j]) / 1000) + (i == 1) / 2 + (i == 2)
        s = 3 * amp[j]
        ax.scatter(x, y, s=s, c=c[i], linewidths=0)
ax.set_xlim(-2, 2)
ax.set_ylim(0, 50)
ax.set_yticks(np.arange(0, 51, 10))
ax.set_ylabel("Distance [km]")
ax.set_xticks([-2, -1, 0, 1, 2])
ax.set_xticklabels(["-2:00", "-1:00", "0:00", "1:00", "2:00"])
fig.tight_layout()
fig.savefig("../figs/branch_association.pdf")
