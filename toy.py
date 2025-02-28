import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import jax
from src.models import KuramotoOscillators
from mne.time_frequency.tfr import tfr_array_morlet
from tqdm import tqdm
from hoi.core import get_mi
from frites.core import copnorm_nd

## Simulation parameters

ntrials = 200
fsamp = 10000
time = np.arange(-0.5, 1, 1 / fsamp)
Npoints = len(time)
Nareas = 3
C = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]]).T
beta = 1

f = 40  # np.linspace(20, 60, Nareas)[::-1]  # Node natural frequency in Hz

muee = 10

Iext = np.zeros((Nareas, Npoints))
Iext[0, (time > 0) & (time < 0.1)] = 10

data = []
for n in tqdm(range(ntrials)):
    temp, dt_save = KuramotoOscillators(
        C * muee, f, -5, fsamp, beta, Npoints, None, (n + 1) * Iext
    )
    data += [temp]

data = np.stack(data)
# Output the shapes of data and datah for verification
data.shape

### Convert to xarray

data = xr.DataArray(
    data[..., ::15],
    dims=("trials", "roi", "times"),
    coords=((np.arange(ntrials)) + 1, ["x", "y", "z"], time[::15]),
)

## Plot

data = data.sel(times=slice(-0.2, 2))


z_data = (data - data.mean("times")) / data.std("times")
for i in range(Nareas):
    plt.plot(z_data[0].times, z_data[-1].values[i] + (i * 3))


plt.show()

##

plt.subplot(1, 2, 1)
CC = np.corrcoef(data[0].real)
plt.imshow(CC, cmap="hot_r", vmin=0, vmax=0.5, origin="lower")
plt.yticks(range(Nareas), data.roi.values)
plt.xticks(range(Nareas), data.roi.values, rotation=90)
plt.colorbar()

plt.subplot(1, 2, 2)
CC = np.corrcoef(data[-1].real)
plt.imshow(CC, cmap="hot_r", vmin=0, vmax=0.5, origin="lower")
plt.yticks(range(Nareas), data.roi.values)
plt.xticks(range(Nareas), data.roi.values, rotation=90)
plt.colorbar()
plt.show()

### Decompose in time-frequency domain

freqs = np.linspace(0.3, 80, 30)

S = tfr_array_morlet(
    data.values.real,
    fsamp / 15,
    freqs,
    freqs / 7,
    output="complex",
    n_jobs=1,
    zero_mean=False,
    verbose=True,
)

S = xr.DataArray(
    S,
    dims=("trials", "roi", "freqs", "times"),
    coords={"freqs": freqs, "times": data.times.values, "roi": data.roi},
)

### Compute phase and amplitude terms

# Define the function to compute MI using HOI and JAX
mi_fcn = get_mi("gc")

# vectorize the function to first and second dimension
gcmi = jax.vmap(jax.vmap(mi_fcn, in_axes=0), in_axes=0)

# Select data for nodes
x = S.sel(roi=["x"]).data.squeeze()
y = S.sel(roi=["y"]).data.squeeze()
z = S.sel(roi=["z"]).data.squeeze()

# Edge activity (with and without normalisation)
e1 = x * np.conj(y)
e2 = y * np.conj(z)
e3 = x * np.conj(z)

# Real and Imag parts pf edge activity
e1r, e1i = np.real(e1), np.imag(e1)
e2r, e2i = np.real(e2), np.imag(e2)
e3r, e3i = np.real(e3), np.imag(e3)

# Stack complex values
E1 = np.stack((e1r, e1i), axis=1)
E2 = np.stack((e2r, e2i), axis=1)
E3 = np.stack((e3r, e3i), axis=1)
E12 = np.stack((e1r, e1i, e2r, e2i), axis=1)
E23 = np.stack((e2r, e2i, e3r, e3i), axis=1)
E123 = np.stack((e1r, e1i, e2r, e2i, e3r, e3i), axis=1)

# Swap axis 0 with 3 and 1 with 2
E1 = np.moveaxis(E1, [0, 1], [-1, -2])
E2 = np.moveaxis(E2, [0, 1], [-1, -2])
E3 = np.moveaxis(E3, [0, 1], [-1, -2])
E12 = np.moveaxis(E12, [0, 1], [-1, -2])
E23 = np.moveaxis(E23, [0, 1], [-1, -2])
E123 = np.moveaxis(E123, [0, 1], [-1, -2])

# Stims across trials
stim = data.trials.values
stim = np.expand_dims(stim, axis=(0, 1))
stim = np.tile(stim, (len(freqs), data.sizes["times"], 1, 1))


# Copnorm
E1 = copnorm_nd(E1, axis=-1)
E2 = copnorm_nd(E2, axis=-1)
E3 = copnorm_nd(E3, axis=-1)
E12 = copnorm_nd(E12, axis=-1)
E23 = copnorm_nd(E23, axis=-1)
E123 = copnorm_nd(E123, axis=-1)
stim = copnorm_nd(stim, axis=-1)

# MIF for edge encoding
mif_e1 = gcmi(E1, stim).T
mif_e2 = gcmi(E2, stim).T
mif_e3 = gcmi(E3, stim).T
mif_e12 = gcmi(E12, stim).T
mif_e23 = gcmi(E23, stim).T
mif_e123 = gcmi(E123, stim).T

red = np.minimum(mif_e1, mif_e2)

syn = mif_e123 - np.maximum(mif_e1, mif_e2)

## Plot

times = data.times.values
ax = plt.subplot()
# Plot single-trial and time-frequency MI
mi = xr.DataArray(mif_e3, dims=("times", "freqs"), coords=(times, freqs))
# Plot trial-average time-frequency MI
mi.plot(x="times", y="freqs", cmap="turbo", vmin=0)
# mi.plot.title('ciao')
ax.set_title(
    "Coherence between node 2 and 3 using Frequency-domain MI (Kuramoto Chain)"
)
ax.set_xlabel("Time (samples)")
ax.set_ylabel("Frequency (Hz)")
plt.show()

ax = plt.subplot()
# Plot single-trial and time-frequency MI
mi = xr.DataArray(mif_e123, dims=("times", "freqs"), coords=(times, freqs))
# Plot trial-average time-frequency MI
mi.plot(x="times", y="freqs", cmap="turbo", vmin=0)
# mi.plot.title('ciao')
ax.set_title("Total MI I(e1, e2; coupling)")
ax.set_xlabel("Time (samples)")
ax.set_ylabel("Frequency (Hz)")
plt.show()

ax = plt.subplot()
# Plot single-trial and time-frequency MI
mi = xr.DataArray(red, dims=("times", "freqs"), coords=(times, freqs))
# Plot trial-average time-frequency MI
mi.plot(x="times", y="freqs", cmap="turbo", vmin=0)
# mi.plot.title('ciao')
ax.set_title("Redundancy")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Frequency (Hz)")
plt.show()

ax = plt.subplot()
# Plot single-trial and time-frequency MI
mi = xr.DataArray(syn, dims=("times", "freqs"), coords=(times, freqs))
# Plot trial-average time-frequency MI
mi.plot(x="times", y="freqs", cmap="turbo", vmin=0)
# mi.plot.title('ciao')
ax.set_title("Synergy")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Frequency (Hz)")
