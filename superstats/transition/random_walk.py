from typing import Tuple, Dict, Any
import numpy as np
from numba import njit, prange

from .transition import Transition, Prior
from superstats.utils.transformations import scaled_sigmoid


@njit(parallel=True, fastmath=True)
def _sample_random_walk(
    local_params: np.ndarray,
    sigma: np.ndarray,
    delta: np.ndarray,
    bounds: Tuple[float, float],
) -> np.ndarray:
    # Generate random walk trajectories with noise and optional drift.
    batch_size, steps = local_params.shape
    lower, upper = bounds

    noise = np.random.randn(batch_size, steps - 1)

    for b in prange(batch_size):
        increments = delta[b] + sigma[b] * noise[b]
        local_params[b, 1:] = local_params[b, 0] + np.cumsum(increments)

        local_params[b, :] = scaled_sigmoid(local_params[b, :], lower, upper)

    return local_params


class RandomWalk(Transition):
    """
    Random walk transition process with optional drift.

    Generates trajectories that evolve as a random walk with Gaussian noise
    and optional deterministic drift, bounded within specified limits.
    """

    def __init__(
        self,
        bounds: Tuple[float, float],
        initial_prior=None,
        sigma: float | Prior = 0.1,
        delta: float | Prior = 0.0,
    ):
        """
        Initialize a random walk transition.

        Parameters
        ----------
        bounds : tuple of float
            Parameter bounds (lower, upper).
        initial_prior : Prior, optional
            Prior distribution for initial parameter values.
        sigma : float or Prior, optional
            Step size (standard deviation of noise). Default is 0.1.
        delta : float or Prior, optional
            Drift term added to each step. Default is 0.0.
        """
        super().__init__(bounds, initial_prior)

        self.hyper_specs = {
            "sigma": sigma,
            "delta": delta,
        }

        self.transition_type = "rw"

    def _expand_to_batch(self, x, batch_size: int):
        # Expand scalar values to batch-sized arrays.
        if np.ndim(x) == 0:
            return np.full(batch_size, x, dtype=self.dtype)
        return x

    def _resolve_hyperparams(self, batch_size: int):
        # Split hyperparameters into sampled vs fixed values.
        hyper_params = {}
        fixed_params = {}

        for name, value in self.hyper_specs.items():
            if isinstance(value, Prior):
                hyper_params[name] = value.sample(batch_size)
            else:
                fixed_params[name] = value

        return hyper_params, fixed_params

    def sample(self, batch_size: int, steps: int) -> Dict[str, Any]:
        """
        Generate random walk parameter trajectories.

        Parameters
        ----------
        batch_size : int
            Number of independent trajectories.
        steps : int
            Number of time steps per trajectory.

        Returns
        -------
        dict
            Dictionary containing:
            - 'local_params': np.ndarray of shape (batch_size, steps)
            - 'hyper_params': dict of sampled hyperparameters
            - 'fixed_params': dict of fixed hyperparameters
        """
        local_params = np.empty((batch_size, steps), dtype=self.dtype)
        local_params[:, 0] = self.initial_prior.sample(batch_size)

        hyper_params, fixed_params = self._resolve_hyperparams(batch_size)

        sigma = self._expand_to_batch(
            hyper_params["sigma"] if "sigma" in hyper_params else fixed_params["sigma"],
            batch_size,
        )

        delta = self._expand_to_batch(
            hyper_params["delta"] if "delta" in hyper_params else fixed_params["delta"],
            batch_size,
        )

        local_params = _sample_random_walk(
            local_params,
            sigma,
            delta,
            self.bounds,
        )

        return {
            "local_params": local_params,
            "hyper_params": hyper_params,
            "fixed_params": fixed_params,
        }