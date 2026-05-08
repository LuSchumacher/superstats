from __future__ import annotations

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
    bounds: np.ndarray,
) -> np.ndarray:

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
    noise = np.random.randn()
    return x + delta + sigma * noise


class RandomWalk(Transition):
    """
    Random walk transition with Gaussian noise and optional drift.
    """

    def __init__(
        self,
        bounds: Tuple[float, float] | None = None,
        initial_prior: Prior | None = None,
        sigma: float | Prior | None = None,
        delta: float | Prior = 0.0,
    ):
        super().__init__(bounds, initial_prior)

        self.hyper_specs = {
            "sigma": sigma,
            "delta": delta,
        }

        self.transition_type = "rw"

    def sample(self, batch_size: int, steps: int) -> Dict[str, Any]:

        local_params = np.empty((batch_size, steps), dtype=self.dtype)
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
        sigma = float(params["sigma"])
        delta = float(params["delta"])
        return _one_step_random_walk(
            x,
            sigma,
            delta,
        )