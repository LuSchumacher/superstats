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
    x: float,
    sigma: float,
    phi: float,
    delta: float,
) -> float:
    noise = np.random.randn()
    return phi * x + delta + sigma * noise


class AutoRegression(Transition):
    """AR(1) autoregressive transition.

    Parameters
    ----------
    bounds : tuple or None
        Lower and upper bounds for the latent state.
    initial_prior : Prior or None
        Prior for the initial latent state.
    sigma : float or Prior or None
        Standard deviation of the innovation noise.
    phi : float or Prior or None
        Autoregressive coefficient.
    delta : float or Prior, optional
        Additive drift term (default 0.0).

    Notes
    -----
    Implements an AR(1): x_t = phi * x_{t-1} + delta + sigma * eps_t.
    """

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
        """
        Draw `batch_size` AR(1) trajectories of length `steps`.

        Parameters
        ----------
        batch_size : int
        steps : int

        Returns
        -------
        dict
            Dictionary with keys ``local_params``, ``hyper_params``,
            and ``fixed_params``.
        """

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

    def sample_one_step(self, x: float, params: Dict[str, Any]) -> float:
        """
        Advance a single AR(1) step.

        Parameters
        ----------
        x : float
            Previous latent state.
        params : dict
            Expect keys ``sigma``, ``phi``, ``delta``.

        Returns
        -------
        float
            Next latent state.
        """

        sigma = float(params["sigma"])
        phi = float(params["phi"])
        delta = float(params["delta"])

        return _one_step_ar1(
            x,
            sigma,
            phi,
            delta,
        )