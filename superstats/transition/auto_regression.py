from typing import Tuple, Dict, Any
import numpy as np
from numba import njit, prange

from .transition import Transition, Prior
from superstats.utils.transformations import scaled_sigmoid


@njit(parallel=True, fastmath=True)
def _sample_ar1(
    local_params: np.ndarray,
    sigma: np.ndarray,
    phi: np.ndarray,
    delta: np.ndarray,
    bounds: Tuple[float, float],
) -> np.ndarray:
    # Generate AR(1) trajectories and apply bounds.
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
        sigma: float | Prior | None = None,
        phi: float | Prior | None = None,
        delta: float | Prior = 0.0,
    ):
        """
        Initialize an AR(1) transition process.

        Parameters
        ----------
        bounds : tuple of float
            Parameter bounds (lower, upper).
        initial_prior : Prior, optional
            Prior for initial values.
        sigma : float or Prior, optional
            Standard deviation of innovations. Default is None.
        phi : float or Prior, optional
            Autocorrelation coefficient. Default is None.
        delta : float or Prior, optional
            Drift component added to each step. Default is 0.0.
        """
        super().__init__(bounds, initial_prior)

        self.hyper_specs = {
            "sigma": sigma,
            "phi": phi,
            "delta": delta,
        }

        self.transition_type = "ar1"

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
            Dictionary containing:
            - 'local_params': np.ndarray of shape (batch_size, steps)
            - 'hyper_params': dict of sampled hyperparameters
            - 'fixed_params': dict of fixed hyperparameters
        """
        local_params = np.empty((batch_size, steps), dtype=self.dtype)
        local_params[:, 0] = self.initial_prior.sample(batch_size)

        hyper_params, fixed_params = self._resolve_hyperparams(batch_size)

        sigma = self._expand_to_batch(
            hyper_params.get("sigma", fixed_params["sigma"]),
            batch_size,
        )

        phi = self._expand_to_batch(
            hyper_params.get("phi", fixed_params["phi"]),
            batch_size,
        )

        delta = self._expand_to_batch(
            hyper_params.get("delta", fixed_params["delta"]),
            batch_size,
        )

        local_params = _sample_ar1(
            local_params,
            sigma,
            phi,
            delta,
            self.bounds,
        )

        return {
            "local_params": local_params,
            "hyper_params": hyper_params,
            "fixed_params": fixed_params,
        }
