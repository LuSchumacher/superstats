"""Lévy-flight transition models."""

from typing import Dict, Any
from collections.abc import Sequence
import numpy as np
from numba import njit, prange

from .transition import StochasticTransition, Prior
from superstats.utils.transformations import scaled_sigmoid


@njit
def _sample_alpha_stable(alpha: float, beta: float, scale: float) -> float:
    """Draw one symmetric-ish alpha-stable variate via Chambers-Mallows-Stuck.

    Parameters
    ----------
    alpha : float
        Stability index in (0, 2]. alpha=2 recovers a Gaussian (with
        scale sqrt(2) relative to std); smaller alpha gives heavier tails.
    beta  : float
        Skewness in [-1, 1]. beta=0 is symmetric.
    scale : float
        Scale parameter (analogous to sigma for the increments).

    Returns
    -------
    x : float - a single alpha-stable increment
    """
    u = (np.random.rand() - 0.5) * np.pi
    w = -np.log(np.random.rand())

    if alpha == 1.0:
        # Special-case alpha == 1 to avoid division by zero.
        x = (2.0 / np.pi) * (
            (np.pi / 2.0 + beta * u) * np.tan(u)
            - beta * np.log((np.pi / 2.0 * w * np.cos(u)) / (np.pi / 2.0 + beta * u))
        )
    else:
        zeta = beta * np.tan(np.pi * alpha / 2.0)
        xi = np.arctan(zeta) / alpha
        x = (
            (1.0 + zeta * zeta) ** (1.0 / (2.0 * alpha))
            * np.sin(alpha * (u + xi))
            / (np.cos(u) ** (1.0 / alpha))
            * (np.cos(u - alpha * (u + xi)) / w) ** ((1.0 - alpha) / alpha)
        )

    return scale * x


@njit(parallel=True, fastmath=True)
def _sample_levy_flight(
    local_params: np.ndarray,
    sigma: np.ndarray,
    delta: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    bounds: np.ndarray,
) -> np.ndarray:
    """Vectorized Lévy-flight rollout across a batch, filled in place.

    Parameters
    ----------
    local_params : np.ndarray of shape (batch_size, steps)
        Pre-allocated trajectory array; `local_params[:, 0]` must already
        hold the initial state. Overwritten in place with the full rollout.
    sigma        : np.ndarray of shape (batch_size,)
        Scale of the alpha-stable increments, per trajectory.
    delta        : np.ndarray of shape (batch_size,)
        Additive drift term, per trajectory.
    alpha        : np.ndarray of shape (batch_size,)
        Stability index in (0, 2], per trajectory.
    beta         : np.ndarray of shape (batch_size,)
        Skewness in [-1, 1], per trajectory.
    bounds       : np.ndarray of shape (2,)
        (lower, upper) bounds passed to `scaled_sigmoid`.

    Returns
    -------
    local_params : np.ndarray of shape (batch_size, steps) - the same
        array, filled with the bounded Lévy-flight rollout
    """
    batch_size, steps = local_params.shape
    lower, upper = bounds[0], bounds[1]

    for b in prange(batch_size):
        for t in range(1, steps):
            increment = delta[b] + _sample_alpha_stable(alpha[b], beta[b], sigma[b])
            local_params[b, t] = local_params[b, t - 1] + increment
        local_params[b, :] = scaled_sigmoid(local_params[b, :], lower, upper)

    return local_params


@njit
def _one_step_levy_flight(
    x: float,
    sigma: float,
    delta: float,
    alpha: float,
    beta: float,
) -> float:
    """Advance a single Lévy-flight state by one step (scalar, JIT-compiled).

    Parameters
    ----------
    x     : float
        Previous latent state.
    sigma : float
        Scale of the alpha-stable increment.
    delta : float
        Additive drift term.
    alpha : float
        Stability index in (0, 2].
    beta  : float
        Skewness in [-1, 1].

    Returns
    -------
    x_next : float - the next latent state
    """
    return x + delta + _sample_alpha_stable(alpha, beta, sigma)


class LevyFlight(StochasticTransition):
    """Lévy-flight transition with alpha-stable noise and optional drift.

    Like a random walk, but the Gaussian increments are replaced with
    alpha-stable increments, introducing a stability index `alpha` that
    controls tail heaviness. `alpha=2` recovers Gaussian-like behavior;
    smaller `alpha` produces heavy-tailed jumps.

    Parameters
    ----------
    bounds        : tuple or None, optional, default: None
        Lower and upper bounds for the latent state.
    initial_prior : Prior or None, optional, default: None
        Prior for the initial latent state.
    sigma         : float or Prior or None, optional, default: None
        Scale of the alpha-stable increments.
    delta         : float or Prior or None, optional, default: None
        Additive drift term.
    alpha         : float or Prior or None, optional, default: None
        Stability index in (0, 2] controlling tail heaviness.
    beta          : float or Prior or None, optional, default: None
        Skewness in [-1, 1]. Defaults to 0 (symmetric) if left unset.

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
        alpha: float | Prior | None = None,
        beta: float | Prior | None = None,
    ):
        super().__init__(bounds, initial_prior)

        self.hyper_specs = {
            "sigma": sigma,
            "delta": delta,
            "alpha": alpha,
            "beta": beta,
        }

        self.transition_type = "levy"

    def sample(self, batch_size: int, num_steps: int) -> Dict[str, Any]:
        """Draw `batch_size` Lévy-flight trajectories of length `num_steps`.

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

        def _resolve(name, default=None):
            if name in hyper:
                return hyper[name]
            if name in fixed and fixed[name] is not None:
                return np.full(batch_size, fixed[name], dtype=self.dtype)
            if default is not None:
                return np.full(batch_size, default, dtype=self.dtype)
            raise KeyError(f"'{name}' must be provided as a hyperparameter or fixed value.")

        sigma = _resolve("sigma")
        delta = _resolve("delta", default=0.0)
        alpha = _resolve("alpha")
        beta = _resolve("beta", default=0.0)

        local_params = _sample_levy_flight(
            local_params,
            sigma.astype(self.dtype),
            delta.astype(self.dtype),
            alpha.astype(self.dtype),
            beta.astype(self.dtype),
            self.bounds,
        )

        return {
            "local_params": local_params,
            "hyper_params": hyper,
            "fixed_params": fixed,
        }

    def sample_one_step(self, x: float, params: Dict[str, Any]) -> float:
        """Advance a single step of the Lévy flight.

        Parameters
        ----------
        x      : float
            Previous latent state.
        params : dict
            Expected keys: `sigma`, `delta`, `alpha`, `beta`.

        Returns
        -------
        x_next : float - the next latent state
        """
        sigma = float(params["sigma"])
        delta = float(params["delta"])
        alpha = float(params["alpha"])
        beta = float(params.get("beta", 0.0))
        return _one_step_levy_flight(x, sigma, delta, alpha, beta)
