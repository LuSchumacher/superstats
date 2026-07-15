"""Periodic covariance kernels."""

import numpy as np
from numba import njit, prange

from .kernel import Kernel


@njit
def build_abs_dist_mat(num_steps: int) -> np.ndarray:
    """Absolute pairwise distances for `num_steps` evenly spaced points on [0, 1].

    Parameters
    ----------
    num_steps : int
        Number of points in the 1D grid.

    Returns
    -------
    abs_dist : np.ndarray of shape (num_steps, num_steps)
        Absolute distance between point i and point j at [i, j].
    """
    x = np.linspace(0, 1, num_steps)
    diff = x.reshape(-1, 1) - x.reshape(1, -1)
    return np.abs(diff)


@njit(parallel=True, fastmath=True)
def get_periodic_kernel(
    num_steps: int, length_scale: np.ndarray, period: np.ndarray, amplitude: np.ndarray
) -> np.ndarray:
    """Batched periodic kernel construction.

    k(x, x') = amplitude^2 * exp(-2 * sin^2(pi * |x - x'| / period) / length_scale^2)

    Parameters
    ----------
    num_steps : int
        Number of points in the 1D grid.
    length_scale : np.ndarray of shape (batch_size,)
        Smoothness of the repeated shape; smaller values mean a more
        wiggly repeating pattern, larger values a smoother one.
    period : np.ndarray of shape (batch_size,)
        Distance between repetitions, on the [0, 1] grid.
    amplitude : np.ndarray of shape (batch_size,)
        Kernel variance / overall scale per trajectory.

    Returns
    -------
    kernel_mat : np.ndarray of shape (batch_size, num_steps, num_steps)
    """
    batch_size = length_scale.shape[0]
    abs_dist = build_abs_dist_mat(num_steps)
    kernel_mat = np.empty((batch_size, num_steps, num_steps))

    for b in prange(batch_size):
        kernel_mat[b] = (amplitude[b] ** 2) * np.exp(
            -2 * np.sin(np.pi * abs_dist / period[b]) ** 2 / length_scale[b] ** 2
        )

    return kernel_mat


class PeriodicKernel(Kernel):
    """Periodic kernel. Requires hyperparameters `length_scale`, `period`, `amplitude`.

    Models functions that repeat themselves exactly. `period` sets the
    distance between repetitions; `length_scale` controls smoothness
    of the repeated shape, same interpretation as in `RBFKernel`.

    Parameters
    ----------
    name : str, optional
        Prefix for this kernel's hyperparameter names. Leave unset for
        a single periodic kernel, or when combining with a kernel of a
        different type. Required when combining two periodic kernels.
    """

    _local_hyperparam_names = ("length_scale", "period", "amplitude")

    def __init__(self, name: str | None = None):
        super().__init__(name)

    def build(self, num_steps: int, **hyperparams: np.ndarray) -> np.ndarray:
        local = self.local_hyperparams(**hyperparams)
        return get_periodic_kernel(num_steps, local["length_scale"], local["period"], local["amplitude"])
