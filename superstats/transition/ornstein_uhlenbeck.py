from typing import Tuple, Dict, Any
import numpy as np
from numba import njit, prange

from .transition import Transition, Prior
from superstats.utils.transformations import scaled_sigmoid


@njit(parallel=True, fastmath=True)
def _sample_ou(
    local_params: np.ndarray,
    mu: np.ndarray,
    theta: np.ndarray,
    sigma: np.ndarray,
    bounds: Tuple[float, float],
) -> np.ndarray:
    # Generate Ornstein-Uhlenbeck trajectories with discrete dt=1.

    batch_size, steps = local_params.shape
    lower, upper = bounds

    noise = np.random.randn(batch_size, steps - 1)

    for b in prange(batch_size):
        x_prev = local_params[b, 0]

        for t in range(1, steps):
            x_prev = (
                x_prev
                + theta[b] * (mu[b] - x_prev)
                + sigma[b] * noise[b, t - 1]
            )
            local_params[b, t] = x_prev

        local_params[b, :] = scaled_sigmoid(local_params[b, :], lower, upper)

    return local_params


class OrnsteinUhlenbeck(Transition):
    """
    Ornstein-Uhlenbeck process transition.

    dx = theta (mu - x) dt + sigma dW
    """

    def __init__(
        self,
        bounds: Tuple[float, float],
        initial_prior=None,
        sigma: float | Prior | None = None,
        mu: float | Prior | None = None,
        theta: float | Prior | None = None,
    ):
        """
        Initialize an Ornstein-Uhlenbeck transition.

        Parameters
        ----------
        bounds : tuple of float
            Bounds for the process values (lower, upper).
        initial_prior : Prior, optional
            Prior distribution for the initial state.
        sigma : float or Prior, optional
            Volatility parameter. Default is None.
        mu : float or Prior, optional
            Long-term mean value. Default is None.
        theta : float or Prior, optional
            Mean reversion rate. Default is None.
        """
        super().__init__(bounds, initial_prior)

        self.hyper_specs = {
            "sigma": sigma,
            "mu": mu,
            "theta": theta,
        }

        self.transition_type = "ou"

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
                fixed_params[name] = value  # keep scalar

        return hyper_params, fixed_params

    def sample(self, batch_size: int, steps: int) -> Dict[str, Any]:
        """
        Generate Ornstein-Uhlenbeck parameter trajectories.

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

        mu = self._expand_to_batch(
            hyper_params.get("mu", fixed_params["mu"]),
            batch_size,
        )

        theta = self._expand_to_batch(
            hyper_params.get("theta", fixed_params["theta"]),
            batch_size,
        )

        local_params = _sample_ou(
            local_params,
            mu,
            theta,
            sigma,
            self.bounds,
        )

        return {
            "local_params": local_params,
            "hyper_params": hyper_params,
            "fixed_params": fixed_params,
        }
