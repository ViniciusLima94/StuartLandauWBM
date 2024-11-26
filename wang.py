import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import jax
from src.models import KuramotoOscillators
from mne.time_frequency.tfr import tfr_array_morlet
from tqdm import tqdm
from hoi.core import get_mi
from frites.core import copnorm_nd

## Load anatomical data
data = np.load("interareal/markov2014.npy", allow_pickle=True).item()

# Graph parameters
Nareas = 29  # Number of areas
# FLN matrix
flnMat = data["FLN"].T
# Distance matrix
D = data["Distances"] * 1e-3 / 3.5
# Hierarchy values
h = np.squeeze(data["Hierarchy"].T)

eta = 4.0

## Simulation parameters

ntrials = 100
fsamp = 1 / 1e-4
time = np.arange(-2, 5, 1 / fsamp)
beta = 0.001
Npoints = len(time)
# Convert to timesteps
D = (D * fsamp).astype(int)

f = 40  # np.linspace(20, 60, Nareas)[::-1]  # Node natural frequency in Hz

muee = 30
flnMat = (1 + eta * h[:, None]) * flnMat

Iext = np.zeros((Nareas, Npoints))
Iext[0, (time >= 0) & (time <= 0.2)] = 1
CS = np.linspace(0, 0.1, ntrials)

data = []
for n in tqdm(range(ntrials)):
    temp, dt_save = KuramotoOscillators(
        flnMat,
        muee,
        f,
        -5,
        fsamp,
        beta,
        Npoints,
        None,
        CS[n] * Iext,
    )
    data += [temp]

data = np.stack(data)
# Output the shapes of data and datah for verification
data.shape

### Convert to xarray

area_names = [
    "V1",
    "V2",
    "V4",
    "DP",
    "MT",
    "8m",
    "5",
    "8l",
    "TEO",
    "2",
    "F1",
    "STPc",
    "7A",
    "46d",
    "10",
    "9/46v",
    "9/46d",
    "F5",
    "TEpd",
    "PBr",
    "7m",
    "7B",
    "F2",
    "STPi",
    "PROm",
    "F7",
    "8B",
    "STPr",
    "24c",
]

data = xr.DataArray(
    data[..., ::15],
    dims=("trials", "roi", "times"),
    coords=(CS, area_names, time[::15]),
)

## Plot

z_data = (data - data.mean("times")) / data.std("times")
for i in range(Nareas):
    plt.plot(z_data[-1].times, z_data[-1].values[i].real + (i * 3))


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

data = data.sel(times=slice(-0.2, 3))


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
    coords={"freqs": freqs, "times": data.times.values, "roi": area_names},
)

### Compute phase and amplitude terms


def _mi(S, roi_x, roi_y, stim):

    # Define the function to compute MI using HOI and JAX
    mi_fcn = get_mi("gcmi")

    # vectorize the function to first and second dimension
    gcmi = jax.vmap(jax.vmap(mi_fcn, in_axes=0), in_axes=0)

    times, freqs = S.times.values, S.freqs.values
    x = S.sel(roi=[roi_x]).data.squeeze()
    y = S.sel(roi=[roi_y]).data.squeeze()

    edge = x * np.conj(y)
    edge_r, edge_i = np.real(edge), np.imag(edge)

    E1 = np.stack((edge_r, edge_i), axis=1)
    E1 = np.moveaxis(E1, [0, 1], [-1, -2])

    # Stims across trials
    stim = data.trials.values
    stim = np.expand_dims(stim, axis=(0, 1))
    stim = np.tile(stim, (len(freqs), data.sizes["times"], 1, 1))

    E1 = copnorm_nd(E1, axis=-1)
    stim = copnorm_nd(stim, axis=-1)

    mi = gcmi(E1, stim).T

    return xr.DataArray(mi, dims=("times", "freqs"), coords=(times, freqs))


pairs = np.stack([[[0] * 28], [range(1, 29)]], axis=1).squeeze().T

rois = S.roi.values
stim = data.trials.values

out = []
for i, j in tqdm(pairs):
    out += [_mi(S, rois[i], rois[j], stim)]


plt.figure(figsize=(40, 40))

for pos, _out in enumerate(out):
    plt.subplot(6, 5, pos + 1)
    i, j = pairs[pos]
    _out.T.plot(cmap="turbo", vmin=0, vmax=1)
    plt.title(f"{rois[i]}-{rois[j]}")
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.ylabel("freq [Hz]", fontsize=9)
    plt.xlabel("time [s]", fontsize=9)
plt.tight_layout()
plt.savefig("mi_cond.png")
