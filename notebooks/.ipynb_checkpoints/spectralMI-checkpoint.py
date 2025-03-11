from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from frites.conn import conn_io
from frites.io import check_attrs, logger, set_log_level
from frites.utils import parallel_func
from hoi.core import get_mi
from mne.time_frequency.tfr import tfr_array_morlet
from tqdm import tqdm
from jax.scipy.special import ndtri


def ctransform(x):
    """Copula transformation (empirical CDF).

    Parameters
    ----------
    x : array_like
        Array of data. The trial axis should be the last one

    Returns
    -------
    xr : array_like
        Empirical CDF value along the last axis of x. Data is ranked and scaled
        within [0 1] (open interval)
    """
    xr = jnp.argsort(jnp.argsort(x)).astype(float)
    xr += 1.0
    xr /= float(xr.shape[-1] + 1)
    return xr


def copnorm_1d(x):
    """Copula normalization for a single vector.

    Parameters
    ----------
    x : array_like
        Array of data of shape (n_epochs,)

    Returns
    -------
    cx : array_like
        Standard normal samples with the same empirical CDF value as the input.
    """
    # assert isinstance(x, jnp.ndarray) and (x.ndim == 1)
    return ndtri(ctransform(x))


def copnorm_nd(x, axis=-1):
    """Copula normalization for a multidimentional array.

    Parameters
    ----------
    x : array_like
        Array of data
    axis : int | -1
        Epoch (or trial) axis. By default, the last axis is considered

    Returns
    -------
    cx : array_like
        Standard normal samples with the same empirical CDF value as the input.
    """
    # assert isinstance(x, jnp.ndarray) and (x.ndim >= 1)
    return jnp.apply_along_axis(copnorm_1d, axis, x)

# Define the function to compute MI using HOI and JAX
mi_fcn = get_mi("gc")

# vectorize the function to first and second dimension
gcmi = jax.vmap(jax.vmap(mi_fcn, in_axes=0), in_axes=0)


def _phase_amplitude(w, foi_idx, x_s, x_t, stim):
    """Compute the MI for each pair"""
    # w = jnp.asarray(w)

    @partial(jax.vmap, in_axes=(0, 0))
    def _pairwise(w_x, w_y):

        edge = w[:, w_x, :, :] * jnp.conj(w[:, w_y, :, :])
        edge_r, edge_i = np.real(edge), np.imag(edge)

        E1 = jnp.stack((edge_r, edge_i), axis=1)
        E1 = jnp.moveaxis(E1, [0, 1], [-1, -2])
        E1 = copnorm_nd(E1, axis=-1)

        return gcmi(E1, stim)

    return _pairwise(x_s, x_t)


def spectralMI(
    data,
    y,
    freqs=None,
    roi=None,
    times=None,
    sfreq=None,
    foi=None,
    n_cycles=7,
    decim=1,
    mode="phase_amplitude",
    block_size=None,
    n_jobs=-1,
    verbose=None,
    dtype=np.float32,
    **kw_links,
):
    """
    Compute pairwise spectral mutual information (MI) between data signals 
    across specific frequency bands and time points.

    Parameters
    ----------
    data : array-like or xarray.DataArray
        Input signal data. Expected to have dimensions (channels, samples, trials).
    y : array-like
        Stimulus or trial labels to compute MI with respect to.
    freqs : array-like, optional
        List of frequencies (Hz) for time-frequency decomposition.
    roi : list of str, optional
        Regions of interest (channel labels).
    times : array-like, optional
        Time points corresponding to the signal data.
    sfreq : float, optional
        Sampling frequency of the data (Hz).
    foi : list of float, optional
        Frequencies of interest for phase-amplitude coupling analysis.
    n_cycles : int, optional
        Number of cycles for Morlet wavelet decomposition. Default is 7.
    decim : int, optional
        Temporal decimation factor to downsample the time axis. Default is 1 (no decimation).
    mode : {'amplitude', 'phase', 'phase_amplitude'}, optional
        Mode of MI computation. Determines which components of the signal 
        are used for MI analysis:
        - 'phase': only phase information
        - 'phase_amplitude': combination of phase and amplitude
        Default is 'phase_amplitude'.
    block_size : int, optional
        Number of channel pairs processed per iteration. Speeds up computations 
        by splitting the data into blocks. Default is None.
    n_jobs : int, optional
        Number of parallel jobs to use for computations. Default is -1 (use all available CPUs).
    verbose : bool or None, optional
        Whether to display logging information during computation. Default is None.
    dtype : data-type, optional
        Data type for the output MI array. Default is np.float32.
    **kw_links : dict, optional
        Additional parameters for input-output handling, e.g., directed or net-related settings.

    Returns
    -------
    conn : xarray.DataArray
        Spectral mutual information matrix with dimensions:
        - 'roi': Regions of interest (channel pairs)
        - 'freqs': Frequencies of interest
        - 'times': Time points
        The MI values are computed for each pair of signals, frequency, and time point.

    Notes
    -----
    - The function uses Morlet wavelet decomposition for time-frequency representation 
      of the input signals.
    - Phase and/or amplitude components of the signals are extracted based on the `mode` parameter.
    - Mutual information is computed between the transformed signal components 
      and stimulus/trial labels (`y`).
    """  
    assert mode in ["phase", "phase_amplitude"]

    # _________________________________ INPUTS ________________________________
    # inputs conversion
    kw_links.update({"directed": False, "net": False})
    data, cfg = conn_io(
        data,
        times=times,
        roi=roi,
        agg_ch=False,
        win_sample=None,
        block_size=block_size,
        sfreq=sfreq,
        freqs=freqs,
        foi=foi,
        verbose=verbose,
        name=f"Spectral MI",
        kw_links=kw_links,
    )

    # extract variables
    x, trials, attrs = data.data, data["y"].data, cfg["attrs"]
    times, n_trials = data["times"].data, len(trials)
    x_s, x_t, roi_p = cfg["x_s"], cfg["x_t"], cfg["roi_p"]
    sfreq = cfg["sfreq"]
    freqs, _, foi_idx = cfg["freqs"], cfg["need_foi"], cfg["foi_idx"]
    f_vec = cfg["f_vec"]
    n_pairs, n_freqs = len(x_s), len(freqs)

    x_s = jnp.atleast_1d(x_s)
    x_t = jnp.atleast_1d(x_t)
    
    # temporal decimation
    times = times[::decim]

    # define arguments for parallel computing
    mesg = f"Estimating pairwise spectral MI for trials %s"
    kw_para = dict(n_jobs=n_jobs, verbose=verbose, total=n_pairs)

    # show info
    logger.info(
        f"Computing pairwise spectral MI (n_pairs={n_pairs}, "
        f"n_freqs={n_freqs}, decim={decim})"
    )

    # Compute for blocks of channel pairs
    indices = np.arange(n_pairs)
    if isinstance(block_size, int):
        indices = np.array_split(indices, block_size)
    print(indices)

    # reshape stimuli
    y = np.expand_dims(y, axis=(0, 1))
    y = np.tile(y, (len(freqs), len(times), 1, 1))
    y = jnp.asarray(copnorm_nd(y, axis=-1))

    conn = np.zeros((n_pairs, len(f_vec), len(times)), dtype=dtype)
    dims = ("roi", "freqs", "times")
    coords = (roi_p, f_vec, times)

    # --------------------------- TIME-FREQUENCY --------------------------
    # time-frequency decomposition
    w = tfr_array_morlet(
        x,
        sfreq,
        freqs,
        n_cycles=n_cycles,
        decim=decim,
        n_jobs=n_jobs,
        output="complex",
    )

    w = jnp.asarray(w)

    if mode == "phase":
        w = w / jnp.sqrt(jnp.abs(w))

    for pr in tqdm(indices):
        conn[pr, ...] = _phase_amplitude(w, foi_idx, x_s[pr], x_t[pr], y)

    # configuration
    cfg = dict(sfreq=sfreq, mode=mode, n_cycles=n_cycles, decim=decim)

    # conversion
    conn = xr.DataArray(
        conn, dims=dims, coords=coords, attrs=check_attrs({**attrs, **cfg})
    )

    return conn