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
    x: float,
    mu: float,
    theta: float,
    sigma: float,
) -> float:
    noise = np.random.randn()
    return x + theta * (mu - x) + sigma * noise


class OrnsteinUhlenbeck(Transition):
    """Ornstein-Uhlenbeck mean-reverting transition.

    Parameters
    ----------
    bounds : tuple or None
        Lower and upper bounds for the latent state.
    initial_prior : Prior or None
        Prior for the initial latent state.
    sigma : float or Prior or None
        Diffusion scale.
    mu : float or Prior or None
        Long-run mean to revert towards.
    theta : float or Prior or None
        Mean-reversion speed.
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

        self.transition_type = "ou"

    def sample(self, batch_size: int, num_steps: int) -> Dict[str, Any]:
        """
        Draw `batch_size` Ornstein-Uhlenbeck trajectories of length `num_steps`.

        Parameters
        ----------
        batch_size : int
        num_steps : int

        Returns
        -------
        dict
            Dictionary with keys ``local_params``, ``hyper_params``,
            and ``fixed_params``.
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
        """
        Advance a single OU step.

        Parameters
        ----------
        x : float
            Previous latent state.
        params : dict
            Expect keys ``mu``, ``theta``, ``sigma``.

        Returns
        -------
        float
            Next latent state.
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
