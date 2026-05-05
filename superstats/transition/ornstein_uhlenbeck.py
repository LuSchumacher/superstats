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
    bounds: np.ndarray,
) -> np.ndarray:

    batch_size, steps = local_params.shape
    lower, upper = bounds[0], bounds[1]

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


@njit
def _one_step_ou(
    x: np.ndarray,
    mu: np.ndarray,
    theta: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:

    noise = np.random.randn(x.shape[0])
    return x + theta * (mu - x) + sigma * noise


class OrnsteinUhlenbeck(Transition):

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

        self.transition_type = "ou"

    def sample(self, batch_size: int, steps: int) -> Dict[str, Any]:

        local_params = np.empty((batch_size, steps), dtype=self.dtype)
        local_params[:, 0] = self.initial_prior.sample(batch_size).astype(self.dtype)

        hyper, fixed = self._resolve_hyperparams(batch_size)

        # -------------------------
        # SAFE resolution (batch arrays guaranteed)
        # -------------------------

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

    def sample_one_step(self, x: np.ndarray, params: Dict[str, Any]) -> np.ndarray:

        mu = np.asarray(params["mu"], dtype=self.dtype)
        theta = np.asarray(params["theta"], dtype=self.dtype)
        sigma = np.asarray(params["sigma"], dtype=self.dtype)

        return _one_step_ou(
            x,
            mu,
            theta,
            sigma,
        )