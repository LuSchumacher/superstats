from typing import Tuple, Dict, Any
import numpy as np
from numba import njit, prange

from .transition import Transition
from superstats.utils.transformations import scaled_sigmoid


@njit(parallel=True, fastmath=True)
def _sample_random_walk(
    local_params: np.ndarray,
    sigma: np.ndarray,
    delta: np.ndarray,
    bounds: Tuple[float, float],
) -> np.ndarray:
    """
    Generate random walk trajectories with noise and optional drift.

    Computes cumulative increments and applies sigmoid transformation to
    enforce bounds.

    Parameters
    ----------
    local_params : np.ndarray
        Array of shape (batch_size, steps) to store trajectories.
    sigma : np.ndarray
        Step sizes of shape (batch_size,).
    delta : np.ndarray
        Drift terms of shape (batch_size,).
    bounds : tuple of float
        Parameter bounds (lower, upper).

    Returns
    -------
    np.ndarray
        Trajectories of shape (batch_size, steps) bounded by bounds.
    """
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
    Gaussian random walk (with optional drift via delta).
    """

    def __init__(
        self,
        bounds: Tuple[float, float],
        initial_prior=None,
        sigma: float | None = None,
        delta: float | None = 0.0,
    ):
        """
        Initialize a random walk transition.

        Parameters
        ----------
        bounds : tuple of float
            Parameter bounds (lower, upper).
        initial_prior : Prior, optional
            Prior for initial values.
        sigma : float, optional
            Standard deviation of increments (step size).
        delta : float, optional
            Drift component added to each increment. Default: 0.0.
        """
        super().__init__(bounds, initial_prior)

        self.global_params = {
            "sigma": sigma,
            "delta": delta,
        }

        self.transition_type = "rw"

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
            Contains 'local_params' (trajectories), 'global_params' (sigma, delta),
            and 'infer_mask' (which hyperparameters are stochastic).
        """
        local_params = np.empty((batch_size, steps), dtype=self.dtype)
        local_params[:, 0] = self.initial_prior.sample(batch_size)
        global_params, infer_mask = self.sample_global_params(batch_size)

        local_params = _sample_random_walk(
            local_params,
            global_params["sigma"],
            global_params["delta"],
            self.bounds,
        )

        return {
            "local_params": local_params,
            "global_params": global_params,
            "infer_mask": infer_mask,
        }
