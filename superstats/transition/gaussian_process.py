"""Gaussian-process transition models."""

from typing import Any, Dict, Sequence, Literal
import numpy as np
from numba import njit, prange

from .transition import Transition, Prior, ParamSpec
from .kernel import Kernel, resolve_kernel
from superstats.utils.transformations import scaled_sigmoid


@njit(parallel=True, fastmath=True)
def sample_gaussian_process(
    local_params: np.ndarray, start: np.ndarray, kernel_mat: np.ndarray, bounds: np.ndarray, noise=1e-6
) -> np.ndarray:
    """Vectorized Gaussian process rollout across a batch, filled in place.

    Parameters
    ----------
    local_params : np.ndarray of shape (batch_size, steps)
        Pre-allocated output array; filled in place with the bounded
        rollout.
    start     : np.ndarray of shape (batch_size,)
        Initial state per trajectory, used as the GP sample mean.
    kernel_mat : np.ndarray of shape (batch_size, steps, steps)
        Covariance kernel matrix, per trajectory.
    bounds    : np.ndarray of shape (2,)
        (lower, upper) bounds passed to `scaled_sigmoid`.
    noise     : float, default 1e-6
        Jitter added to the kernel diagonal for numerical stability
        during Cholesky decomposition.

    Returns
    -------
    local_params : np.ndarray of shape (batch_size, steps) - the same
        array, filled with the bounded Gaussian process rollout
    """
    batch_size, num_steps, _ = kernel_mat.shape
    lower, upper = bounds[0], bounds[1]

    jitter = noise * np.eye(num_steps)

    for b in prange(batch_size):
        cholesky = np.linalg.cholesky(kernel_mat[b] + jitter)
        sample = start[b] + np.dot(cholesky, np.random.randn(num_steps))
        local_params[b] = scaled_sigmoid(sample, lower, upper)

    return local_params


class GaussianProcess(Transition):
    """Gaussian process transition model.

    Draws each trajectory as a GP sample with mean `start` (from
    `initial_prior`) and covariance from `kernel`, then squashes the
    result into `bounds` via a scaled sigmoid.

    Parameters
    ----------
    kernel        : {"rbf", "linear"} or Kernel, default "rbf"
        Kernel used to build the covariance matrix. Either a registered
        name or a `Kernel` instance, including composites built via
        `+`/`*` (e.g. `RBFKernel(name="trend") + RBFKernel(name="local")`).
    kernel_params : dict, optional
        Maps kernel hyperparameter names (see `kernel.hyperparam_names`,
        e.g. `"length_scale"`, `"amplitude"` for the default RBF
        kernel) to a `Prior` (sampled per-batch) or a fixed float. Any
        name not given here falls back to `DEFAULT_HYPER_PRIORS`, same
        as `RandomWalk`'s `sigma`/`delta`. Combining two kernels of the
        same type requires an explicit `name=` on each (see `Kernel`),
        which prefixes their hyperparameter names accordingly.
    bounds        : tuple or None, optional, default: None
        Lower and upper bounds for the latent state.
    initial_prior : Prior or None, optional, default: None
        Prior for the initial latent state.

    Notes
    -----
    The `sample` method returns a dict with keys `local_params`,
    `hyper_params` and `fixed_params`, matching `RandomWalk`.
    """

    def __init__(
        self,
        kernel: Literal["rbf", "linear", "periodic"] | Kernel = "rbf",
        kernel_params: Dict[str, ParamSpec] | None = None,
        bounds: Sequence[float] | None = None,
        initial_prior: Prior | None = None,
    ):
        super().__init__(bounds, initial_prior)

        self.kernel = resolve_kernel(kernel)
        kernel_params = kernel_params or {}

        unknown = set(kernel_params) - set(self.kernel.hyperparam_names)
        if unknown:
            raise ValueError(
                f"Unknown hyperparameter(s) {unknown} for kernel '{kernel}'; "
                f"expected a subset of {self.kernel.hyperparam_names}"
            )

        self.hyper_specs = {name: kernel_params.get(name) for name in self.kernel.hyperparam_names}

        self.transition_type = "gp"

    def sample(self, batch_size: int, num_steps: int) -> Dict[str, Any]:
        """Draw `batch_size` Gaussian process trajectories of length `num_steps`.

        Parameters
        ----------
        batch_size : int
            Number of independent trajectories to draw.
        num_steps  : int
            Number of time steps per trajectory.

        Returns
        -------
        result : dict - dictionary with keys `local_params`,
            `hyper_params`, and `fixed_params`
        """
        hyper, fixed = self._resolve_hyperparams(batch_size)

        kernel_args = {
            name: hyper[name] if name in hyper else np.full(batch_size, fixed[name], dtype=self.dtype)
            for name in self.kernel.hyperparam_names
        }
        kernel_mat = self.kernel.build(num_steps, **kernel_args)

        start = self.initial_prior.sample(batch_size).astype(self.dtype)

        local_params = np.empty((batch_size, num_steps), dtype=self.dtype)
        local_params = sample_gaussian_process(
            local_params,
            start,
            kernel_mat,
            self.bounds,
        )

        return {
            "local_params": local_params,
            "hyper_params": hyper,
            "fixed_params": fixed,
        }
