"""Random-walk transition models."""

from typing import Dict, Any
from collections.abc import Sequence
import numpy as np
from numba import njit, prange

from .transition import Transition, Prior
from superstats.utils.transformations import scaled_sigmoid


@njit(parallel=True, fastmath=True)
def _sample_random_walk(
    local_params: np.ndarray,
    sigma: np.ndarray,
    delta: np.ndarray,
    bounds: np.ndarray,
) -> np.ndarray:
    """Vectorized random-walk rollout across a batch, filled in place.

    Parameters
    ----------
    local_params : np.ndarray of shape (batch_size, steps)
        Pre-allocated trajectory array; `local_params[:, 0]` must already
        hold the initial state. Overwritten in place with the full rollout.
    sigma        : np.ndarray of shape (batch_size,)
        Standard deviation of the Gaussian increments, per trajectory.
    delta        : np.ndarray of shape (batch_size,)
        Additive drift term, per trajectory.
    bounds       : np.ndarray of shape (2,)
        (lower, upper) bounds passed to `scaled_sigmoid`.

    Returns
    -------
    local_params : np.ndarray of shape (batch_size, steps) - the same
        array, filled with the bounded random-walk rollout
    """
    batch_size, steps = local_params.shape
    lower, upper = bounds[0], bounds[1]

    noise = np.random.randn(batch_size, steps - 1)

    for b in prange(batch_size):
        increments = delta[b] + sigma[b] * noise[b]
        local_params[b, 1:] = local_params[b, 0] + np.cumsum(increments)
        local_params[b, :] = scaled_sigmoid(local_params[b, :], lower, upper)

    return local_params


@njit
def _one_step_random_walk(
    x: float,
    sigma: float,
    delta: float,
) -> float:
    """Advance a single random-walk state by one step (scalar, JIT-compiled).

    Parameters
    ----------
    x     : float
        Previous latent state.
    sigma : float
        Standard deviation of the Gaussian increment.
    delta : float
        Additive drift term.

    Returns
    -------
    x_next : float - the next latent state
    """
    noise = np.random.randn()
    return x + delta + sigma * noise


class RandomWalk(Transition):
    """Random walk transition with Gaussian noise and optional drift.

    Parameters
    ----------
    bounds        : tuple or None, optional, default: None
        Lower and upper bounds for the latent state.
    initial_prior : Prior or None, optional, default: None
        Prior for the initial latent state.
    sigma         : float or Prior or None, optional, default: None
        Standard deviation of the Gaussian increments.
    delta         : float or Prior, optional, default: 0.0
        Additive drift term.

    Notes
    -----
    The `sample` method returns a dict with keys `local_params`,
    `hyper_params` and `fixed_params`. Use `sample_one_step` to advance
    a single time-step given numeric params.
    """

    def __init__(
        self,
        bounds: Sequence[float, float] | None = None,
        initial_prior: Prior | None = None,
        sigma: float | Prior | None = None,
        delta: float | Prior | None = None,
    ):
        super().__init__(bounds, initial_prior)

        self.hyper_specs = {
            "sigma": sigma,
            "delta": delta,
        }

        self.transition_type = "rw"

    def sample(self, batch_size: int, num_steps: int) -> Dict[str, Any]:
        """Draw `batch_size` random-walk trajectories of length `num_steps`.

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

        if "delta" in hyper:
            delta = hyper["delta"]
        else:
            delta = np.full(batch_size, fixed["delta"], dtype=self.dtype)

        local_params = _sample_random_walk(
            local_params,
            sigma.astype(self.dtype),
            delta.astype(self.dtype),
            self.bounds,
        )

        return {
            "local_params": local_params,
            "hyper_params": hyper,
            "fixed_params": fixed,
        }

    def sample_one_step(self, x: float, params: Dict[str, Any]) -> float:
        """Advance a single step of the random walk.

        Parameters
        ----------
        x      : float
            Previous latent state.
        params : dict
            Expected keys: `sigma`, `delta`.

        Returns
        -------
        x_next : float - the next latent state
        """
        sigma = float(params["sigma"])
        delta = float(params["delta"])
        return _one_step_random_walk(
            x,
            sigma,
            delta,
        )
