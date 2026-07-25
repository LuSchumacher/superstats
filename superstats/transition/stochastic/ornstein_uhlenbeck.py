"""Ornstein-Uhlenbeck transition models."""

from typing import Tuple, Dict, Any
import numpy as np
from numba import njit, prange

from stochastic import StochasticTransition, Prior
from superstats.utils.transformations import scaled_sigmoid


@njit(parallel=True, fastmath=True)
def _sample_ou(
    local_params: np.ndarray,
    mu: np.ndarray,
    theta: np.ndarray,
    sigma: np.ndarray,
    bounds: np.ndarray,
) -> np.ndarray:
    """Vectorized Ornstein-Uhlenbeck rollout across a batch, filled in place.

    Parameters
    ----------
    local_params : np.ndarray of shape (batch_size, steps)
        Pre-allocated trajectory array; `local_params[:, 0]` must already
        hold the initial state. Overwritten in place with the full rollout.
    mu           : np.ndarray of shape (batch_size,)
        Long-run mean to revert towards, per trajectory.
    theta        : np.ndarray of shape (batch_size,)
        Mean-reversion speed, per trajectory.
    sigma        : np.ndarray of shape (batch_size,)
        Diffusion scale, per trajectory.
    bounds       : np.ndarray of shape (2,)
        (lower, upper) bounds passed to `scaled_sigmoid`.

    Returns
    -------
    local_params : np.ndarray of shape (batch_size, steps) - the same
        array, filled with the bounded OU rollout
    """
    batch_size, steps = local_params.shape
    lower, upper = bounds[0], bounds[1]

    noise = np.random.randn(batch_size, steps - 1)

    for b in prange(batch_size):
        x_prev = local_params[b, 0]

        for t in range(1, steps):
            x_prev = x_prev + theta[b] * (mu[b] - x_prev) + sigma[b] * noise[b, t - 1]
            local_params[b, t] = x_prev

        local_params[b, :] = scaled_sigmoid(local_params[b, :], lower, upper)

    return local_params


@njit
def _one_step_ou(
    x: float,
    mu: float,
    theta: float,
    sigma: float,
) -> float:
    """Advance a single Ornstein-Uhlenbeck state by one step (scalar, JIT-compiled).

    Parameters
    ----------
    x     : float
        Previous latent state.
    mu    : float
        Long-run mean to revert towards.
    theta : float
        Mean-reversion speed.
    sigma : float
        Diffusion scale.

    Returns
    -------
    x_next : float - the next latent state
    """
    noise = np.random.randn()
    return x + theta * (mu - x) + sigma * noise


class OrnsteinUhlenbeck(StochasticTransition):
    """Ornstein-Uhlenbeck mean-reverting transition.

    Parameters
    ----------
    bounds        : tuple or None, optional, default: None
        Lower and upper bounds for the latent state.
    initial_prior : Prior or None, optional, default: None
        Prior for the initial latent state.
    sigma         : float or Prior or None, optional, default: None
        Diffusion scale.
    mu            : float or Prior or None, optional, default: None
        Long-run mean to revert towards.
    theta         : float or Prior or None, optional, default: None
        Mean-reversion speed.

    Notes
    -----
    Implements an OU process:
    x_t = x_{t-1} + theta * (mu - x_{t-1}) + sigma * eps_t.
    """

    def __init__(
        self,
        bounds: Tuple[float, float] | None = None,
        initial_prior: Prior | None = None,
        sigma: float | Prior | None = None,
        mu: float | Prior | None = None,
        theta: float | Prior | None = None,
    ):
        super().__init__(bounds, initial_prior)

        self.hyper_specs = {
            "sigma": sigma,
            "mu": mu,
            "theta": theta,
        }

        self.transition_name = "ou"

    def sample(self, batch_size: int, num_steps: int) -> Dict[str, Any]:
        """Draw `batch_size` Ornstein-Uhlenbeck trajectories of length `num_steps`.

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
        local_params = np.empty((batch_size, num_steps), dtype=self.dtype)
        local_params[:, 0] = self.initial_prior.sample(batch_size).astype(self.dtype)

        hyper, fixed = self._resolve_hyperparams(batch_size)

        if "sigma" in hyper:
            sigma = hyper["sigma"]
        else:
            sigma = np.full(batch_size, fixed["sigma"], dtype=self.dtype)

        if "mu" in hyper:
            mu = hyper["mu"]
        else:
            mu = np.full(batch_size, fixed["mu"], dtype=self.dtype)

        if "theta" in hyper:
            theta = hyper["theta"]
        else:
            theta = np.full(batch_size, fixed["theta"], dtype=self.dtype)

        local_params = _sample_ou(
            local_params,
            mu.astype(self.dtype),
            theta.astype(self.dtype),
            sigma.astype(self.dtype),
            self.bounds,
        )

        return {
            "local_params": local_params,
            "hyper_params": hyper,
            "fixed_params": fixed,
        }

    def sample_one_step(self, x: float, params: Dict[str, Any]) -> float:
        """Advance a single OU step.

        Parameters
        ----------
        x      : float
            Previous latent state.
        params : dict
            Expected keys: `mu`, `theta`, `sigma`.

        Returns
        -------
        x_next : float - the next latent state
        """
        mu = float(params["mu"])
        theta = float(params["theta"])
        sigma = float(params["sigma"])
        return _one_step_ou(
            x,
            mu,
            theta,
            sigma,
        )
