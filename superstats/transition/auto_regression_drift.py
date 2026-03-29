from typing import Tuple, Union
import numpy as np
from numba import njit, prange

from .transition import Transition
from superstats.utils.transformations import scaled_sigmoid
from superstats.prior import Prior

from .transition import Transition

@njit(parallel=True, fastmath=True)
def sample_auto_regression_drift(
    local_params: np.ndarray,
    phi: np.ndarray,
    delta: np.ndarray,
    sigma: np.ndarray,
    bounds: Tuple[float, float],
) -> np.ndarray:
    batch_size, steps = local_params.shape
    lower, upper = bounds

    noise = np.random.randn(batch_size, steps - 1)

    for b in prange(batch_size):
        # Random walk case (phi == 1) → use cumulative sum
        if phi[b] == 1.0:
            increments = delta[b] + sigma[b] * noise[b]
            local_params[b, 1:] = local_params[b, 0] + np.cumsum(increments)
        # General AR(1)
        else:
            x_prev = local_params[b, 0]
            for t in range(1, steps):
                x_prev = phi[b] * x_prev + delta[b] + sigma[b] * noise[b, t - 1]
                local_params[b, t] = x_prev

        local_params[b, :] = scaled_sigmoid(local_params[b, :], lower, upper)

    return local_params


class AutoRegressionDrift(Transition):
    """General AR(1) with drift stochastic transition."""

    def __init__(
        self,
        bounds: Tuple[float, float],
        initial_prior: Prior = None,
        phi_prior: Prior | float = None,
        delta_prior: Prior | float = None,
        sigma_prior: Prior | float = None,
    ):
        super().__init__(bounds, initial_prior)

        if phi_prior is None:
            self.phi_prior = Prior("beta", a=5.0, b=1.0)
        else:
            self.phi_prior = phi_prior

        if delta_prior is None:
            self.delta_prior = Prior("normal", loc=0.0, scale=0.05)
        else:
            self.delta_prior = delta_prior

        if sigma_prior is None:
            self.sigma_prior = Prior("halfnormal", scale=0.1)
        else:
            self.sigma_prior = sigma_prior

    def _draw_param(self, prior_or_value, batch_size):
        """Utility: draw samples or broadcast constant."""
        if isinstance(prior_or_value, Prior):
            return prior_or_value.sample(batch_size)
        else:
            return np.full(batch_size, prior_or_value, dtype=np.float32)

    def sample(self, batch_size: int, steps: int) -> dict:
        """Sample trajectories."""

        local_params = np.empty((batch_size, steps), dtype=np.float32)
        local_params[:, 0] = self.initial_prior.sample(batch_size)

        phi = self._draw_param(self.phi_prior, batch_size)
        delta = self._draw_param(self.delta_prior, batch_size)
        sigma = self._draw_param(self.sigma_prior, batch_size)

        local_params = sample_auto_regression_drift(
            local_params,
            phi,
            delta,
            sigma,
            self.bounds
        )

        return {
            "local_params": local_params,
            "phi": phi,
            "delta": delta,
            "sigma": sigma
        }