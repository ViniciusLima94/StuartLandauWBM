import numpy as np
from src.models import KuramotoOscillators
from frites.utils import parallel_func

## Load anatomical data
data = np.load("interareal/markov2014.npy", allow_pickle=True).item()

# Graph parameters
Nareas = 29  # Number of areas
# FLN matrix
flnMat = data["FLN"]
# Distance matrix
D = data["Distances"] * 1e-3 / 3.5
# Hierarchy values
h = np.squeeze(data["Hierarchy"].T)

eta = 4.0

## Simulation parameters

ntrials = 100
fsamp = 10000
time = np.arange(0, 5, 1 / fsamp)
Npoints = len(time)
# Convert to timesteps
D = (D * fsamp).astype(int)


def _simulate(f=40, muee=1, beta=1, flnMat=None, h=None):

    flnMat = muee * flnMat

    Iext = np.zeros((Nareas, Npoints))

    data, dt_save = KuramotoOscillators(
        flnMat, f, -0.5, fsamp, beta, Npoints, None, Iext
    )

    data = np.stack(data)

    theta = np.angle(data)
    order_parameter = np.exp(1j * theta).mean(axis=0)
    r = (order_parameter * np.conj(order_parameter)).mean().real

    return r


muee_vec = np.linspace(0, 1, 30)
beta_vec = np.linspace(0, 1, 30)

pars = np.array(np.meshgrid(muee_vec, beta_vec)).T.reshape(-1, 2)

# define the function to compute in parallel
parallel, p_fun = parallel_func(_simulate, n_jobs=4, verbose=True, total=pars.shape[0])
# Compute the single trial coherence
r = parallel(p_fun(muee=muee, beta=beta, flnMat=flnMat, h=h) for muee, beta in pars)

# r = np.zeros((muee_vec.shape[0], beta_vec.shape[0]))
#
# for i, muee in tqdm(enumerate(muee_vec)):
#     for j, beta in enumerate(beta_vec):
#         r[i, j] = _simulate(muee=muee, beta=beta, flnMat=flnMat, h=h)
