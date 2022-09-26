import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

## Wrapped Cauchy

# parameters
n = 1_000_000  # number of samples
snr = 100  # in amplitude
R = 1 - 1 / snr  # mean resultant length

# generate signals
s = snr * np.random.randn(n)  # signal
p = s + np.random.randn(n)  # pressure
vx = s + np.random.randn(n)  # x velocity
vy = np.random.randn(n)  # y velocity
ia = p * (vx + 1j * vy)  # active acoustic intensity
az = np.arctan2(ia.imag, ia.real)  # azimuth

# statistical model
phi = np.linspace(-np.pi, np.pi, 3601)
pdf = st.wrapcauchy(R).pdf(phi % (2 * np.pi))

# plot
plt.style.use("../figures.mplstyle")
fig, ax = plt.subplots(constrained_layout=True)
ax.hist(
    az,
    bins=360 * 20,
    range=(-np.pi, np.pi),
    density=True,
    color="grey",
    label=r"$\rm{SNR}^2=100$",
)
ax.plot(
    phi,
    pdf,
    ls="--",
    lw=0.75,
    color="C3",
    label=r"$\mathcal{W}\mathcal{C}(1 - 1/\rm{SNR}^2)$",
)
degmax = 5
deg = np.arange(-degmax, degmax + 1)
ax.set_xlim(-np.deg2rad(degmax), np.deg2rad(degmax))
ax.set_xticks(np.deg2rad(deg))
ax.set_xticklabels(deg)
ax.set_xlabel("Azimuth [°]")
ax.set_ylim(0, 35)
ax.set_ylabel("Density")
ax.legend(loc="upper right")
