import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from src.models import KuramotoOscillators
from tqdm import tqdm
import scipy

## Load anatomical data
red_mat = scipy.io.loadmat("SC_90aal_32HCP.mat")["mat"]

# Graph parameters
N = len(red_mat)  # Number of areas

## Simulation parameters

ntrials = 1
fsamp = 1 / 1e-4
time = np.arange(-2, 5, 1 / fsamp)
beta = 0.001
Npoints = len(time)

f = 40  # np.linspace(20, 60, Nareas)[::-1]  # Node natural frequency in Hz

muee = 10
C = red_mat / np.mean(red_mat[(np.ones((N, N)) - np.eye(N)) > 0])

Iext = np.zeros((N, Npoints))
# Iext[0, (time >= 0) & (time <= 0.2)] = 1

data = []
for n in tqdm(range(ntrials)):
    temp, dt_save = KuramotoOscillators(
        C.T,
        muee,
        f,
        -5.0,
        fsamp,
        beta,
        Npoints,
        None,
        np.linspace(0, 0.1, ntrials)[n] * Iext,
    )
    data += [temp]

data = np.stack(data)
# Output the shapes of data and datah for verification
data.shape

### Convert to xarray

area_names = range(N)

data = xr.DataArray(
    data,
    dims=("trials", "roi", "times"),
    coords=((np.arange(ntrials)) + 1, area_names, time),
)

## Plot

z_data = (data - data.mean("times")) / data.std("times")
for i in range(N):
    plt.plot(z_data[-1].times, z_data[0].values[i].real + (i * 3))


plt.show()
