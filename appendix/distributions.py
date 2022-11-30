"""
Plot figure 9.
"""

import matplotlib.pyplot as plt
import numpy as np
import obsea
import scipy.signal as sp
import scipy.stats as st
from obspy.core import Trace

## Wrapped Cauchy

# parameters
n = 1_000_000  # number of samples
snr = 100**2  # in power
R = 1 - 1 / np.sqrt(snr)  # mean resultant length

# generate signals
s = np.sqrt(snr) * np.random.randn(n)  # signal
taup = s + np.random.randn(n)  # pressure
vx = s + np.random.randn(n)  # x velocity
vy = np.random.randn(n)  # y velocity
ia = taup * (vx + 1j * vy)  # active acoustic intensity
az = np.arctan2(ia.imag, ia.real)  # azimuth

# statistical model
phi = np.linspace(-np.pi, np.pi, 3601)
pdf = st.wrapcauchy(R).pdf(phi % (2 * np.pi))

# plot
plt.style.use("../figures.mplstyle")
fig, axes = plt.subplots(nrows=3, figsize=(3.4, 4), constrained_layout=True)
ax = axes[0]
ax.hist(
    az,
    bins=360 * 20,
    range=(-np.pi, np.pi),
    density=True,
    color="grey",
    label=r"Monte Carlo",
)
ax.plot(
    phi,
    pdf,
    ls="--",
    lw=0.75,
    color="C3",
    label=r"Model",
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


## Cepstral statistics


# parameters
n = 10_000_000
nperseg = 1024
step = nperseg // 2
fs = 50.0
lag = 128
alpha = 0.5
phase = np.pi / 4

# generate signals
s = np.random.randn(n)
t = (np.arange(32 + 1) - 32 // 2) / fs
yI, yQ = sp.gausspulse(t, fc=15.0, bw=2 / 3, retquad=True)

w_direct = yI
w_echo = alpha * np.real(np.cos(phase) * yI + np.sin(phase) * yQ)
w_direct /= np.sqrt(np.sum(w_direct**2))
w_echo /= np.sqrt(np.sum(w_direct**2))

x_direct = np.convolve(s, w_direct, "valid")[lag:-lag]
x_echo = np.convolve(np.roll(s, lag), w_echo, "valid")[lag:-lag]

noise = np.random.randn(x_direct.shape[0])

tr = Trace(x_direct + x_echo + noise, header=dict(channel="BDH", sampling_rate=fs))
tf = obsea.stft(tr, nperseg, step, None)
c = obsea.cepstrogram(tf)

tau = c["quefrency"].values
tau_s = tau[-1]
sigma = (0.625 + 0.2 * np.cos(np.pi * tau / tau_s)) / np.sqrt(nperseg)
mu = np.zeros_like(tau)
mu[lag] = 0.183
mu = np.convolve(mu, w_echo, "same")


ax = axes[1]
ax.hist(
    c[3 * lag // 2],
    bins=101,
    range=(-0.2, 0.2),
    density=True,
    alpha=0.75,
    label="noise",
)
ax.hist(c[lag], bins=101, range=(-0.2, 0.2), density=True, alpha=0.75, label="signal")
taup = np.linspace(-0.2, 0.2, 101)
pdf = st.norm(scale=sigma[3 * lag // 2]).pdf(taup)
ax.plot(taup, pdf, color="black", ls="--", label="model")
pdf = st.norm(scale=sigma[lag], loc=np.mean(c[lag])).pdf(taup)
ax.plot(taup, pdf, color="black", ls="--")
ax.set_xlabel("Value")
ax.set_ylabel("Density")
ax.legend(loc="upper left")

ax = axes[2]
c[1:-1].mean("time").plot(ax=ax, label=r"$\mu_x$")
c[1:-1].std("time").plot(ax=ax, label=r"$\sigma_x$")
ax.plot(tau, mu, ls="--", color="black", label=r"$\mu_{model}$")
ax.plot(tau, sigma, ls="--", color="black", label=r"$\sigma_{model}$")
ax.set_ylim(-0.1, 0.1)
ax.set_xlim(0, tau_s)
ax.set_xlabel("Quefrency [s]")
ax.set_ylabel("Value")
ax.legend(loc="lower right", ncol=2)

fig.savefig("distribution.svg")
