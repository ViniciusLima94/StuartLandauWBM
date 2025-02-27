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


def simulate(
    A: np.ndarray,
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

    N, A, omegas, phases_history, dt, a = _set_nodes(A, f, fs, a)

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

    @jax.jit
    def _loop(carry, t):

        phases_history = carry

        phases_t = phases_history.squeeze().copy()

        phase_differences = phases_t - phases_history

        exp_phi = gain * jnp.exp(1j * (jnp.angle(phases_t) + phi)) + offset

        # Input to each node
        Input = (A * phase_differences).sum(axis=1) + Iext[:, t] * exp_phi * jnp.sqrt(
            jnp.abs(phases_t)
        )

        phases_history = phases_history.at[:, 0].set(
            phases_t
            + dt * _ode(phases_t, a, omegas)
            + Input
            + eta * randn(size=(N,), seed=seed + t)
            + eta * 1j * randn(size=(N,), seed=seed + t + 2 * t)
        )

        carry = jax.lax.reshape(phases_history, (N, 1))
        return carry, phases_history

    _, phases = jax.lax.scan(_loop, (phases_history), times)

    return phases[::decim]


def simulate2(
    A: np.ndarray,
    g: float,
    f: float,
    a: float,
    fs: float,
    eta: float,
    T: float,
    Iext: np.ndarray = None,
    alpha: float = 1,
    seed: int = 0,
    device: str = "cpu",
    decim: int = 1,
):

    assert device in ["cpu", "gpu"]

    jax.config.update("jax_platform_name", device)

    N, A, omegas, phases_history, dt, a = _set_nodes(A, f, fs, a)

    if Iext is not None:
        Iext = jnp.asarray(Iext)  # Assure it is a jax ndarray

    r = jnp.abs(phases_history).astype(jnp.float32)
    theta = jnp.angle(phases_history).astype(jnp.float32)

    times = np.arange(T, dtype=int)  # Time array

    # Scale witgh dt to avoid doing it evert time-step
    # A = g * A * dt
    # Nodes strength
    S = A.sum(axis=1)
    eta = eta * jnp.sqrt(dt)
    Iext_r = Iext * alpha
    Iext_0 = Iext * (1 - alpha)

    @jax.jit
    def _loop(carry, t):

        r, theta = carry
        r_t, theta_t = r.squeeze().copy(), theta.squeeze().copy()

        phase_differences = -(theta - theta_t)
        amp_ratios = r / r_t

        Isyn_r = (A * jnp.tile(r_t, (N, 1)) * jnp.cos(phase_differences)).sum(1)
        Isyn_0 = (A * amp_ratios * jnp.sin(phase_differences)).sum(1)

        r = r.at[:, 0].set(
            r_t
            + dt * drdt(r_t, a, g, S, Isyn_r, Iext_r)
            + eta * randn(size=(N,), seed=seed + t)
        )

        theta = theta.at[:, 0].set(
            theta_t
            + dt * d0dt(omegas, g, Isyn_0, Iext_0)
            + eta * randn(size=(N,), seed=seed + t)
        )

        return (r, theta), r * jnp.exp(1j * theta)

    _, z = jax.lax.scan(_loop, (r, theta), times)

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
