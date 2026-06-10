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
    """Random walk transition with Gaussian noise and optional drift.

    Parameters
    ----------
    bounds : tuple or None
        Lower and upper bounds applied to latent state (default uses
        package-wide defaults).
    initial_prior : Prior or None
        Prior for the initial latent state.
    sigma : float or Prior or None
        Standard deviation of the Gaussian increments.
    delta : float or Prior, optional
        Additive drift term (default 0.0).

    Notes
    -----
    The `sample` method returns a dict with keys ``local_params``,
    ``hyper_params`` and ``fixed_params``. Use `sample_one_step` to
    advance a single time-step given numeric params.
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
        """
        Draw `batch_size` random-walk trajectories of length `steps`.

        Parameters
        ----------
        batch_size : int
            Number of trajectories to draw.
        steps : int
            Number of time points (including initial state).

        Returns
        -------
        dict
            Contains ``local_params`` (ndarray of shape ``(batch_size, steps)``),
            ``hyper_params`` (sampled per-batch hyperparameters) and
            ``fixed_params`` (fixed scalar hyperparameters).
        """

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
        """
        Advance a single step of the random walk.

        Parameters
        ----------
        x : float
            Previous latent state.
        params : dict
            Mapping of parameter names to numeric values (expects
            ``sigma`` and ``delta``).

        Returns
        -------
        float
            Next latent state.
        """

        sigma = float(params["sigma"])
        delta = float(params["delta"])
        return _one_step_random_walk(
            x,
            sigma,
            delta,
        )