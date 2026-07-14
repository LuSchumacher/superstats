"""Linear covariance kernels."""

import numpy as np
from numba import njit, prange

from .kernel import Kernel


@njit(parallel=True, fastmath=True)
def get_linear_kernel(num_steps: int, variance: np.ndarray) -> np.ndarray:
    """Batched linear kernel construction.

    Parameters
    ----------
    num_steps : int
        Number of points in the 1D grid.
    variance : np.ndarray of shape (batch_size,)
        Scale of the kernel per trajectory.

    Returns
    -------
    kernel_mat : np.ndarray of shape (batch_size, num_steps, num_steps)
    """
    batch_size = variance.shape[0]
    x = np.linspace(0, 1, num_steps)
    outer = np.outer(x, x)
    kernel_mat = np.empty((batch_size, num_steps, num_steps))

    for b in prange(batch_size):
        kernel_mat[b] = variance[b] * outer

    return kernel_mat


class LinearKernel(Kernel):
    """Linear kernel. Requires hyperparameter `variance`.

    Parameters
    ----------
    name : str, optional
        Prefix for this kernel's hyperparameter name. Leave unset for
        a single linear kernel, or when combining with a kernel of a
        different type. Required when combining two linear kernels.
    """

    _local_hyperparam_names = ("variance",)

    def __init__(self, name: str | None = None):
        super().__init__(name)

    def build(self, num_steps: int, **hyperparams: np.ndarray) -> np.ndarray:
        local = self.local_hyperparams(**hyperparams)
        return get_linear_kernel(num_steps, local["variance"])
