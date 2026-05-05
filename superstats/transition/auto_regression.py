from __future__ import annotations

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
    bounds: np.ndarray,
) -> np.ndarray:

    batch_size, steps = local_params.shape
    lower, upper = bounds[0], bounds[1]

    noise = np.random.randn(batch_size, steps - 1)

    for b in prange(batch_size):
        x_prev = local_params[b, 0]

        for t in range(1, steps):
            x_prev = phi[b] * x_prev + delta[b] + sigma[b] * noise[b, t - 1]
            local_params[b, t] = x_prev

        local_params[b, :] = scaled_sigmoid(local_params[b, :], lower, upper)

    return local_params


@njit
def _one_step_ar1(
    x: np.ndarray,
    sigma: np.ndarray,
    phi: np.ndarray,
    delta: np.ndarray,
) -> np.ndarray:

    noise = np.random.randn(x.shape[0])
    return phi * x + delta + sigma * noise


class AutoRegression(Transition):

    def __init__(
        self,
        bounds: Tuple[float, float] | None = None,
        initial_prior: Prior | None = None,
        sigma: float | Prior | None = None,
        phi: float | Prior | None = None,
        delta: float | Prior = 0.0,
    ):
        super().__init__(bounds, initial_prior)

        self.hyper_specs = {
            "sigma": sigma,
            "phi": phi,
            "delta": delta,
        }

        self.transition_type = "ar1"

    def sample(self, batch_size: int, steps: int) -> Dict[str, Any]:

        local_params = np.empty((batch_size, steps), dtype=self.dtype)
        local_params[:, 0] = self.initial_prior.sample(batch_size).astype(self.dtype)

        hyper, fixed = self._resolve_hyperparams(batch_size)

        # -------------------------
        # SAFE parameter resolution
        # -------------------------

        if "sigma" in hyper:
            sigma = hyper["sigma"]
        else:
            sigma = np.full(batch_size, fixed["sigma"], dtype=self.dtype)

        if "phi" in hyper:
            phi = hyper["phi"]
        else:
            phi = np.full(batch_size, fixed["phi"], dtype=self.dtype)

        if "delta" in hyper:
            delta = hyper["delta"]
        else:
            delta = np.full(batch_size, fixed["delta"], dtype=self.dtype)

        local_params = _sample_ar1(
            local_params,
            sigma.astype(self.dtype),
            phi.astype(self.dtype),
            delta.astype(self.dtype),
            self.bounds,
        )

        return {
            "local_params": local_params,
            "hyper_params": hyper,
            "fixed_params": fixed,
        }

    def sample_one_step(self, x: np.ndarray, params: Dict[str, Any]) -> np.ndarray:

        sigma = np.asarray(params["sigma"], dtype=self.dtype)
        phi = np.asarray(params["phi"], dtype=self.dtype)
        delta = np.asarray(params["delta"], dtype=self.dtype)

        return _one_step_ar1(
            x,
            sigma,
            phi,
            delta,
        )