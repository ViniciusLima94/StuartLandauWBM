import jax
import jax.numpy as jnp
import numpy as np
from .models_setup import _set_nodes, _set_nodes_delayed
from .utils import randn
from functools import partial

jax.config.update("jax_enable_x64", True)

"""
FUNCTIONS LOOP MIGHT BE COMPILED WITH JAX SCAN LATER
"""


# @partial(jax.vmap, in_axes=(0, 0, 0))
def _ode(Z: np.complex128, a: float, w: float):
    return Z * (a + 1j * w - jnp.abs(Z * Z))


def drdt(r, a, g, S, Isyn, Iext):
    return (a - g * S - r * r) * r + g * Isyn + Iext


def d0dt(omega, g, Isyn, Iext):
    return omega + g * Isyn + Iext


def _check_params(Iext: jnp.ndarray, N: int):
    if isinstance(Iext, (int, float)):
        return jnp.ones((1, N)) * Iext  # Assure it is a jax ndarray
    elif Iext is None:
        return jnp.zeros((1, N))
    return jnp.asarray(Iext)


def simulate_hopf(
    A: np.ndarray,
    g: float,
    f: float,
    a: float,
    fs: float,
    eta: float,
    T: float,
    seed: int = 0,
    device: str = "cpu",
    decim: int = 1,
):

    assert device in ["cpu", "gpu"]

    jax.config.update("jax_platform_name", device)

    N, A, omegas, phases_history, dt, a = _set_nodes(A, f, fs, a)

    g = _check_params(g, T).squeeze()
    # print(g)

    times = np.arange(T, dtype=int)  # Time array

    # Scale with dt to avoid doing it evert time-step
    A = A * dt
    eta = eta * jnp.sqrt(dt)

    @jax.jit
    def _loop(carry, t):

        phases_history = carry

        phases_t = phases_history.squeeze().copy()

        phase_differences = phases_t - phases_history

        # Input to each node
        Isyn = (g[t] * A * phase_differences).sum(axis=1)

        phases_history = phases_history.at[:, 0].set(
            phases_t
            + dt * _ode(phases_t, a, omegas)
            + Isyn
            + eta * randn(size=(N,), seed=seed + t)
            + eta * 1j * randn(size=(N,), seed=seed + t + 2 * t)
        )

        carry = jax.lax.reshape(phases_history, (N, 1))
        return carry, phases_history

    _, phases = jax.lax.scan(_loop, (phases_history), times)

    return phases[::decim].squeeze().T


def simulate(
    A: np.ndarray,
    g_r: float,
    g_0: float,
    f: float,
    a: float,
    fs: float,
    eta: float,
    T: float,
    Iext_r: np.ndarray = None,
    Iext_0: np.ndarray = None,
    seed: int = 0,
    model: str = "kuramoto",
    device: str = "cpu",
    decim: int = 1,
):

    assert device in ["cpu", "gpu"]

    jax.config.update("jax_platform_name", device)

    N, A, omegas, phases_history, dt, a = _set_nodes(A, f, fs, a)

    # Check inputs
    Iext_r = _check_params(Iext_r, N)
    Iext_0 = _check_params(Iext_0, N)
    g_r = _check_params(g_r, N)
    g_0 = _check_params(g_0, N)

    # Initialize carry variables
    r = jnp.abs(phases_history)
    theta = jnp.angle(phases_history)
    # Time array
    times = np.arange(T, dtype=int)

    # Nodes strength
    S = A.sum(axis=1)
    eta = eta * jnp.sqrt(dt)

    ##########################################################################
    # Loop functions for simulate dynamics
    ##########################################################################
    @jax.jit
    def _loop_hopf(carry, t):

        r, theta = carry
        r_t, theta_t = r.squeeze().copy(), theta.squeeze().copy()

        phase_differences = -(theta - theta_t)
        amp_ratios = r / r_t

        Isyn_r = (A * jnp.tile(r_t, (N, 1)) * jnp.cos(phase_differences)).sum(1)
        Isyn_0 = (A * amp_ratios * jnp.sin(phase_differences)).sum(1)

        r = r.at[:, 0].set(
            r_t
            + dt * drdt(r_t, a, g_r[t], S, Isyn_r, Iext_r[:, t])
            + eta * randn(size=(N,), seed=seed + t)
        )

        theta = theta.at[:, 0].set(
            theta_t
            + dt * d0dt(omegas, g_0[t], Isyn_0, Iext_0[:, t])
            + eta * randn(size=(N,), seed=seed + t + 2 * t)
        )

        return (r, theta), r * jnp.exp(1j * theta)

    @jax.jit
    def _loop_kuramoto(carry, t):

        theta = carry
        theta_t = theta.squeeze().copy()

        phase_differences = -(theta - theta_t)

        Isyn_0 = (A * jnp.sin(phase_differences)).sum(1)

        theta = theta.at[:, 0].set(
            theta_t
            + dt * d0dt(omegas, g_0[t], Isyn_0, Iext_0[:, t])
            + eta * randn(size=(N,), seed=seed + t)
        )

        return (theta), theta

    if model == "kuramoto":
        carry = theta
    else:
        carry = (r, theta)

    _, z = jax.lax.scan(eval(f"_loop_{model}"), carry, times)

    z = z.squeeze().T

    if model == "kuramoto":
        Fourier = jnp.fft.fft(jnp.sin(z), n=T, axis=1)
        z = jnp.real(jnp.fft.ifft(Fourier))

    return z[..., ::decim]


def simulate3(
    A: np.ndarray,
    g: np.ndarray,
    f: float,
    a: float,
    fs: float,
    eta: float,
    T: float,
    seed: int = 0,
    device: str = "cpu",
    decim: int = 1,
):

    assert device in ["cpu", "gpu"]

    jax.config.update("jax_platform_name", device)

    N, A, omegas, phases_history, dt, a = _set_nodes(A, f, fs, a)

    if isinstance(g, (int, float)):
        g = g * jnp.ones(T)

    times = np.arange(T, dtype=int)  # Time array

    # Scale with dt to avoid doing it evert time-step
    A = A * dt
    eta = eta * jnp.sqrt(dt)

    @jax.jit
    def _loop(carry, t):

        phases_history = carry

        phases_t = phases_history.squeeze().copy()

        phase_differences = phases_t - phases_history

        # Input to each node
        Input = (A * phase_differences).sum(axis=1)

        phases_history = phases_history.at[:, 0].set(
            phases_t
            + dt * _ode_hopf(phases_t, a, omegas)
            + g[t] * Input
            + eta * randn(size=(N,), seed=seed + t)
            + eta * 1j * randn(size=(N,), seed=seed + t + 2 * t)
        )

        carry = jax.lax.reshape(phases_history, (N, 1))
        return carry, phases_history

    # @jax.jit
    # def _loop(carry, t):

    #    r, theta = carry
    #    r_t, theta_t = r.squeeze().copy(), theta.squeeze().copy()

    #    phase_differences = -(theta - theta_t)
    #    amp_ratios = r / r_t

    #    Isyn_r = (A * jnp.tile(r_t, (N, 1)) * jnp.cos(phase_differences)).sum(1)
    #    Isyn_0 = (A * amp_ratios * jnp.sin(phase_differences)).sum(1)

    #    r = r.at[:, 0].set(
    #        r_t
    #        + dt * drdt(r_t, a, g[t], S, Isyn_r, 0)
    #        + eta * randn(size=(N,), seed=seed + t)
    #    )

    #    theta = theta.at[:, 0].set(
    #        theta_t
    #        + dt * d0dt(omegas, g[t], Isyn_0, 0)
    #        + eta * randn(size=(N,), seed=seed + t + 2 * t)
    #    )

    #    return (r, theta), r * jnp.exp(1j * theta)

    _, z = jax.lax.scan(_loop, (phases_history), times)

    return z[::decim].squeeze().T


def simulate_delayed(
    A: np.ndarray,
    D: np.ndarray,
    g: float,
    f: float,
    a: float,
    fs: float,
    eta: float,
    T: float,
    Iext: np.ndarray = None,
    seed: int = 0,
    device: str = "cpu",
    decim: int = 1,
    stim_mode: str = "amp",
):

    assert stim_mode in ["amp", "phase", "both"]
    assert device in ["cpu", "gpu"]

    jax.config.update("jax_platform_name", device)

    N, A, D, omegas, phases_history, dt, a = _set_nodes_delayed(A, D, f, fs, a)

    if Iext is None:
        Iext = jnp.zeros((N, T))
    else:
        Iext = jnp.asarray(Iext)  # Assure it is a jax ndarray

    # Stim parameters
    gain = 0
    phi = 0
    offset = 1

    if stim_mode == "amp":
        gain = 1
        offset = 0
    elif stim_mode == "phase":
        gain = 1
        phi = np.pi / 2
        offset = 0

    times = np.arange(T, dtype=int)  # Time array

    # Scale with dt to avoid doing it evert time-step
    A = g * A * dt
    eta = eta * jnp.sqrt(dt)
    Iext = Iext * dt

    nodes = jnp.arange(N)

    @jax.jit
    def _loop_delayed(carry, t):

        phases_history = carry

        phases_t = phases_history[:, -1].copy()

        @partial(jax.vmap, in_axes=(0, 0))
        def _return_phase_differences(n, d):
            return phases_history[np.indices(d.shape)[0], d - 1] - phases_t[n]

        phase_differences = _return_phase_differences(nodes, D)

        # phase_differences = np.stack(
        #    [_return_phase_differences(n, d) for n, d in enumerate(D)]
        # )

        exp_phi = gain * jnp.exp(1j * (jnp.angle(phases_t) + phi)) + offset

        # Input to each node
        Input = (A * phase_differences).sum(axis=1) + Iext[:, t] * exp_phi

        phases_history = phases_history.at[:, :-1].set(phases_history[:, 1:])

        phases_history = phases_history.at[:, -1].set(
            phases_t
            + dt * _ode(phases_t, a, omegas)
            + Input
            + eta * randn(size=(N,), seed=seed + t)
            + eta * 1j * randn(size=(N,), seed=seed + t + 2 * t)
        )

        carry = phases_history  # jax.lax.reshape(phases_history, (N, max_delay))
        return carry, phases_history[:, -1]

    _, phases = jax.lax.scan(_loop_delayed, (phases_history), times)

    return phases[::decim]
