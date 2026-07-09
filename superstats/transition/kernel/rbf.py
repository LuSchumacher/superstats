import numpy as np
from numba import njit, prange

from .kernel import Kernel


@njit
def build_sq_dist_mat(num_steps: int) -> np.ndarray:
    """Squared pairwise distances for `num_steps` evenly spaced points on [0, 1].

    Parameters
    ----------
    num_steps : int
        Number of points in the 1D grid.

    Returns
    -------
    sq_dist : np.ndarray of shape (num_steps, num_steps)
        Squared distance between point i and point j at [i, j].
    """
    x = np.linspace(0, 1, num_steps)
    diff = x.reshape(-1, 1) - x.reshape(1, -1)
    return diff**2


@njit(parallel=True, fastmath=True)
def get_rbf_kernel(num_steps: int, length_scale: np.ndarray, amplitude: np.ndarray) -> np.ndarray:
    """Batched RBF kernel construction.

    Parameters
    ----------
    num_steps : int
        Number of points in the 1D grid.
    length_scale : np.ndarray of shape (batch_size,)
        Kernel width per trajectory; larger values mean smoother,
        slower-varying draws.
    amplitude : np.ndarray of shape (batch_size,)
        Kernel variance / overall scale per trajectory.

    Returns
    -------
    kernel_mat : np.ndarray of shape (batch_size, num_steps, num_steps)
    """
    batch_size = length_scale.shape[0]
    sq_dist = build_sq_dist_mat(num_steps)
    kernel_mat = np.empty((batch_size, num_steps, num_steps))

    for b in prange(batch_size):
        kernel_mat[b] = (amplitude[b] ** 2) * np.exp(-sq_dist / (2 * length_scale[b] ** 2))

    return kernel_mat


class RBFKernel(Kernel):
    """RBF kernel. Requires hyperparameters `length_scale`, `amplitude`.

    Parameters
    ----------
    name : str, optional
        Prefix for this kernel's hyperparameter names. Leave unset for
        a single RBF kernel, or when combining with a kernel of a
        different type. Required when combining two RBF kernels, e.g.
        `RBFKernel(name="short") + RBFKernel(name="long")`.
    """

    _local_hyperparam_names = ("length_scale", "amplitude")

    def __init__(self, name: str | None = None):
        super().__init__(name)

    def build(self, num_steps: int, **hyperparams: np.ndarray) -> np.ndarray:
        local = self.local_hyperparams(**hyperparams)
        return get_rbf_kernel(num_steps, local["length_scale"], local["amplitude"])
