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


def simulate(
    A: np.ndarray,
    f: float,
    a: float,
    fs: float,
    eta: float,
    T: float,
    Iext: np.ndarray = None,
    seed: int = 0,
    device: str = "cpu",
):

    assert device in ["cpu", "gpu"]

    jax.config.update("jax_platform_name", device)

    N, A, omegas, phases_history, dt, a = _set_nodes(A, f, fs, a)

    if Iext is None:
        Iext = jnp.zeros((N, T))
    else:
        Iext = jnp.asarray(Iext)  # Assure it is a jax ndarray

    times = np.arange(T, dtype=int)  # Time array

    # Scale with dt to avoid doing it evert time-step
    A = A * dt
    eta = eta * jnp.sqrt(dt)
    Iext = Iext * dt

    @jax.jit
    def _loop(carry, t):

        phases_history = carry

        phases_t = phases_history.squeeze().copy()

        phase_differences = phases_t - phases_history

        # Input to each node
        Input = (A * phase_differences).sum(axis=1) + Iext[:, t] * jnp.exp(
            1j * jnp.angle(phases_t)
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

    return phases


def simulate_delayed(
    A: np.ndarray,
    D: np.ndarray,
    f: float,
    a: float,
    fs: float,
    eta: float,
    T: float,
    Iext: np.ndarray = None,
    seed: int = 0,
    device: str = "cpu",
):

    assert device in ["cpu", "gpu"]

    jax.config.update("jax_platform_name", device)

    N, A, D, omegas, phases_history, dt, a = _set_nodes_delayed(A, D, f, fs, a)

    if Iext is None:
        Iext = jnp.zeros((N, T))
    else:
        Iext = jnp.asarray(Iext)  # Assure it is a jax ndarray

    times = np.arange(T, dtype=int)  # Time array

    # Scale with dt to avoid doing it evert time-step
    A = A * dt
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

        # Input to each node
        Input = (A * phase_differences).sum(axis=1) + Iext[:, t] * jnp.exp(
            1j * jnp.angle(phases_t)
        )

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

    return phases
