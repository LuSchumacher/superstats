from typing import Tuple, Dict, Any
import numpy as np
from numba import njit, prange

from .transition import Transition
from superstats.prior import Prior
from superstats.utils.transformations import scaled_sigmoid


@njit(parallel=True, fastmath=True)
def _sample_ar1(
    local_params: np.ndarray,
    sigma: np.ndarray,
    phi: np.ndarray,
    delta: np.ndarray,
    bounds: Tuple[float, float],
) -> np.ndarray:
    """
    Generate AR(1) process trajectories.

    Computes autoregressive sequences with specified correlation and applies
    sigmoid transformation to enforce bounds.

    Parameters
    ----------
    local_params : np.ndarray
        Array of shape (batch_size, steps) to store trajectories.
    sigma : np.ndarray
        Standard deviations of shape (batch_size,).
    phi : np.ndarray
        Autocorrelation coefficients of shape (batch_size,).
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
        x_prev = local_params[b, 0]

        for t in range(1, steps):
            x_prev = phi[b] * x_prev + delta[b] + sigma[b] * noise[b, t - 1]
            local_params[b, t] = x_prev

        local_params[b, :] = scaled_sigmoid(local_params[b, :], lower, upper)

    return local_params


class AutoRegression(Transition):
    """
    AR(1) process with optional drift.
    """

    def __init__(
        self,
        bounds: Tuple[float, float],
        initial_prior=None,
        sigma: Prior | float | None = None,
        phi: Prior | float | None = None,
        delta: Prior | float | None = 0.0,
    ):
        """
        Initialize an AR(1) transition process.

        Parameters
        ----------
        bounds : tuple of float
            Parameter bounds (lower, upper).
        initial_prior : Prior, optional
            Prior for initial values.
        sigma : float, optional
            Standard deviation of innovations.
        phi : float, optional
            Autocorrelation coefficient.
        delta : float, optional
            Drift component added to each step. default: 0.0.
        """
        super().__init__(bounds, initial_prior)

        self.global_params = {
            "sigma": sigma,
            "phi": phi,
            "delta": delta,
        }

        self.transition_type = "ar1"

    def sample(self, batch_size: int, steps: int) -> Dict[str, Any]:
        """
        Generate AR(1) parameter trajectories.

        Parameters
        ----------
        batch_size : int
            Number of independent trajectories.
        steps : int
            Number of time steps per trajectory.

        Returns
        -------
        dict
            Contains 'local_params' (trajectories), 'global_params' (sigma, phi, delta),
            and 'infer_mask' (which hyperparameters are stochastic).
        """
        local_params = np.empty((batch_size, steps), dtype=self.dtype)
        local_params[:, 0] = self.initial_prior.sample(batch_size)
        global_params, infer_mask = self.sample_global_params(batch_size)

        local_params = _sample_ar1(
            local_params,
            global_params["sigma"],
            global_params["phi"],
            global_params["delta"],
            self.bounds,
        )

        return {
            "local_params": local_params,
            "global_params": global_params,
            "infer_mask": infer_mask,
        }
