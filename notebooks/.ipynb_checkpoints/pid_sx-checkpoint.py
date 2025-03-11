"""Compute PID using shared exclusions."""
# Authors: Christian Ferreyra, chrisferreyra13@gmail.com
# License: Apache 2.0

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import digamma

ANTICHAINS = [
    ((0, 1),),
    ((0,),),
    ((1,),),
    ((0,), (1,)),
]
ATOM_NAMES = ["syn", "uniq_0", "uniq_1", "red"]
MOEBIUS_INVERSION_MATRIX = jnp.array(
    [[1, -1, -1, 1], [0, 1, 0, -1], [0, 0, 1, -1], [0, 0, 0, 1]]
)


@jax.jit
def _cdist_abs(x, y) -> jnp.ndarray:
    """Pairwise abs distances between all samples of x and y."""
    diff = jnp.abs(x.T[:, None, :] - y.T[None])
    return diff


@partial(jax.jit, static_argnums=(1,))
def _get_dist_k(diff, k=4):
    """Return distances to kth neighbour for each sample."""
    _max = jnp.max(diff, axis=-1)
    # indices to the closest points
    closest_points = jnp.argsort(_max, axis=1)
    # the kth neighbour is at index k (ignoring the point itself)
    # distance to the k-th neighbor for each point
    k_neighbours = closest_points[:, : k + 1]
    dist_k = jnp.take_along_axis(_max, k_neighbours, axis=1)
    return dist_k, k_neighbours


@jax.jit
def _get_neighbours(diff, dist_k):
    """Get number of neighbours in a ball of dist_k."""
    if diff.ndim > 3:
        diff = jnp.max(diff, axis=-1)

    if dist_k.ndim == 1:
        dist_k = dist_k[:, None]

    n = (diff < dist_k[:, [-1]]).sum(axis=1, dtype=jnp.int32)

    return n


@jax.jit
def _merge_distances(dist_k, inds):
    """Merge distances and indices."""
    inds = jnp.unique(
        inds, return_index=True, axis=0, size=inds.shape[0], fill_value=-1
    )[1]
    inds = jnp.unique(inds, axis=0, size=inds.shape[0], fill_value=-1)
    return dist_k[inds]


@partial(
    jax.jit,
    static_argnums=(
        2,
        3,
        4,
    ),
)
def get_neighbours_3feats(x, batch=None, k=4, n_x1_feats=1, n_x2_feats=1):
    """Return number of neighbours for 3 features case."""
    # input shape should be (n_features, n_variables, n_samples)
    # considering last feature as target
    # pairwise difference between samples
    if batch is not None:
        # keep batch dimension first
        diff = _cdist_abs(x[..., batch], x)
        n_y = jnp.zeros((len(batch), 4, x.shape[1]))
        n_x = jnp.zeros((len(batch), 4, x.shape[1]))
    else:
        diff = _cdist_abs(x, x)
        n_y = jnp.zeros((x.shape[2], 4, x.shape[1]))
        n_x = jnp.zeros((x.shape[2], 4, x.shape[1]))

    # if variables are multidimensional, we need sum over their features
    diff = jnp.concatenate(
        [
            diff[..., :n_x1_feats].sum(axis=-1, keepdims=True),
            diff[..., n_x1_feats : n_x1_feats + n_x2_feats].sum(axis=-1, keepdims=True),
            diff[..., n_x1_feats + n_x2_feats :].sum(axis=-1, keepdims=True),
        ],
        axis=-1,
    )
    # ---- top set ((0,1),)
    dist_k, _ = _get_dist_k(diff, k=k)
    # ball of radius dist_k over the joint space of the sources
    n_x = n_x.at[:, 0].set(_get_neighbours(diff[..., :2], dist_k))
    n_y = n_y.at[:, 0].set(
        (diff[..., 2] < dist_k[:, [-1]]).sum(axis=1, dtype=jnp.int32)
    )
    # ---- marginal space
    # -- x1
    dist_k, indices_k = _get_dist_k(diff.take(jnp.array([0, 2]), axis=-1), k=k)
    n_x = n_x.at[:, 1].set(_get_neighbours(diff[..., 0], dist_k))
    n_y = n_y.at[:, 1].set(
        (diff[..., 2] < dist_k[:, [-1]]).sum(axis=1, dtype=jnp.int32)
    )
    # -- x2
    dist_k_x2, indices_k_x2 = _get_dist_k(diff[..., 1:], k=k)
    n_x = n_x.at[:, 2].set(_get_neighbours(diff[..., 1], dist_k_x2))
    n_y = n_y.at[:, 2].set(
        (diff[..., 2] < dist_k_x2[:, [-1]]).sum(axis=1, dtype=jnp.int32)
    )
    # ---- bottom set ((0,),(1,))
    dist_k = jnp.hstack([dist_k, dist_k_x2])
    indices_k = jnp.hstack([indices_k, indices_k_x2])
    del dist_k_x2, indices_k_x2

    sort_indices = jnp.argsort(dist_k, axis=1)
    dist_k = jnp.take_along_axis(dist_k, sort_indices, axis=1)
    indices_k = jnp.take_along_axis(indices_k, sort_indices, axis=1)
    del sort_indices
    dist_k = jax.vmap(jax.vmap(_merge_distances), in_axes=-1, out_axes=-1)(
        dist_k, indices_k
    )[:, [k]]
    del indices_k
    n_y = n_y.at[:, 3].set((diff[..., 2] < dist_k).sum(axis=1))
    # it is not necessary to ignore distance to 0, the gate takes care of it
    diff = diff[..., :2] < dist_k[..., None]
    n_x = n_x.at[:, 3].set(
        jnp.logical_or(diff[..., 0], diff[..., 1]).sum(axis=1, dtype=jnp.int32)
    )

    # return n_x1x2, n_x1, n_x2, n_x1_x2, n_y
    return x, jnp.concatenate([n_x, n_y], axis=1)


def estimate_red_3feat(x, k=4, batch_size=None, n_x1_feats=1, n_x2_feats=1):
    """Estimate redundancy for the 3 features case."""
    n_samples = x.shape[2]
    if batch_size is None:
        batch_size = n_samples

    batches = jnp.array(
        [jnp.arange(i, i + batch_size) for i in range(0, n_samples, batch_size)]
    )
    _get_neighbours_fn = partial(
        get_neighbours_3feats,
        k=k,
        n_x1_feats=n_x1_feats,
        n_x2_feats=n_x2_feats,
    )
    _, ns = jax.lax.scan(_get_neighbours_fn, x, batches)
    # ns comes with shape (n_batches, batch_size, 8, n_variables)
    # merge batches as n_samples and reshape it to (8, n_samples, n_variables)
    ns = ns.reshape(-1, 8, x.shape[1]).transpose(1, 0, 2)
    # remove last values after n_samples (they are just a copy of last sample)
    ns = ns[:, :n_samples, :]
    n_y = ns[4:]
    ns = ns[:4]
    I_cap = jnp.zeros(4) if x.ndim == 2 else jnp.zeros((4, x.shape[1]))
    n_samples = float(n_samples)
    for i, _n in enumerate(ns):
        I_cap = I_cap.at[i].set(
            digamma(k)
            + digamma(n_samples)
            - (jnp.mean(digamma(_n), axis=0) + jnp.mean(digamma(n_y[i]), axis=0))
        )

    return I_cap / jnp.log(2)


def compute_pid_sx(x, k, batch_size=None, n_x1_feats=1, n_x2_feats=1):
    """Compute PID atoms using shared exclusions."""
    redundancies = estimate_red_3feat(
        x,
        k=k,
        batch_size=batch_size,
        n_x1_feats=n_x1_feats,
        n_x2_feats=n_x2_feats,
    )
    # get atoms from redundancies
    m = MOEBIUS_INVERSION_MATRIX
    atoms = m @ redundancies

    # format to atom dictionary.
    atom_dict = dict(zip(ATOM_NAMES, atoms))
    return atom_dict


class PIDSx:
    """Estimate PID atoms based on shared exclusions (3 variables).

    Note: Code based on Ehrlich et al., 2024.
    """

    def __init__(self, k=4, batch_size=None):
        """Initialize PID estimator.

        Parameters
        ----------
        k : int, optional
            Number of neighbours for KNN-based redundancy, by default 4.
        batch_size : int, optional
            Number of samples to compute at once in KNN. If None, all samples
            at the same time. By default None.

        Returns
        -------
        PIDSx
            PID estimator.
        """
        self.k = k
        self.batch_size = batch_size

    def _prepare_data(self, x, y):
        """Concatenate and reshape data."""
        self.x = jnp.concatenate([x[:, [i]] for i in range(x.shape[1])] + [y], axis=1)
        if self.x.ndim == 2:
            self.x = self.x[..., None]

            # shape (n_features, n_variables, n_samples)
        self.x = self.x.transpose(1, 2, 0)

    def get_atom(self, atom):
        """Get atom from PID decomposition."""
        if atom not in self.decomp:
            msg = f"{atom} not in PID decomposition."
            raise ValueError(msg)
        return self.decomp[atom]

    def fit(self, x1, x2, y):
        """Compute PID atoms (3 variables)."""
        # reshape data and concatenate
        self._prepare_data(np.hstack([x1, x2]), y)
        n_x1_feats = x1.shape[1]
        n_x2_feats = x2.shape[1]
        # n_y_feats = y.shape[1]
        self.decomp = compute_pid_sx(
            self.x,
            self.k,
            self.batch_size,
            n_x1_feats,
            n_x2_feats,
        )
        return self.decomp
