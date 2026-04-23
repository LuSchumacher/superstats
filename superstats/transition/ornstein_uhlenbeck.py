from typing import Tuple, Dict, Any
import numpy as np
from numba import njit, prange

from .transition import Transition
from superstats.utils.transformations import scaled_sigmoid


@njit(parallel=True, fastmath=True)
def _sample_ou(
    local_params: np.ndarray,
    mu: np.ndarray,
    theta: np.ndarray,
    sigma: np.ndarray,
    dt: float,
    bounds: Tuple[float, float],
) -> np.ndarray:
    """
    Generate Ornstein-Uhlenbeck trajectories.

    Parameters
    ----------
    local_params : np.ndarray
        Array of shape (batch_size, steps) used to store the trajectories.
    mu : np.ndarray
        Long-term mean values of shape (batch_size,).
    theta : np.ndarray
        Mean reversion rates of shape (batch_size,).
    sigma : np.ndarray
        Volatility values of shape (batch_size,).
    dt : float
        Time step size.
    bounds : tuple of float
        Lower and upper bounds for transformed values.

    Returns
    -------
    np.ndarray
        Trajectories of shape (batch_size, steps) bounded by `bounds`.
    """

    batch_size, steps = local_params.shape
    lower, upper = bounds

    noise = np.random.randn(batch_size, steps - 1)

    sqrt_dt = np.sqrt(dt)

    for b in prange(batch_size):
        x_prev = local_params[b, 0]

        for t in range(1, steps):
            x_prev = (
                x_prev
                + theta[b] * (mu[b] - x_prev) * dt
                + sigma[b] * sqrt_dt * noise[b, t - 1]
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
        sigma: float | None = None,
        mu: float | None = None,
        theta: float | None = None,
        dt: float = 1.0,
    ):
        """
        Initialize an Ornstein-Uhlenbeck transition.

        Parameters
        ----------
        bounds : tuple of float
            Bounds for the process values (lower, upper).
        initial_prior : Prior, optional
            Prior distribution for the initial state.
        sigma : float, optional
            Volatility parameter.
        mu : float, optional
            Long-term mean value.
        theta : float, optional
            Mean reversion rate.
        dt : float, optional
            Discrete time increment (default: 1.0).
        """
        super().__init__(bounds, initial_prior)

        self.dt = dt

        self.global_params = {
            "sigma": sigma,
            "mu": mu,
            "theta": theta,
        }

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
            Dictionary containing 'local_params', 'global_params', and 'infer_mask'.
        """

        local_params = np.empty((batch_size, steps), dtype=self.dtype)
        local_params[:, 0] = self.initial_prior.sample(batch_size)

        params, infer = self.sample_global_params(batch_size)

        local_params = _sample_ou(
            local_params,
            params["sigma"],
            params["mu"],
            params["theta"],
            self.dt,
            self.bounds,
        )

        return {
            "local_params": local_params,
            "global_params": params,
            "infer_mask": infer,
        }