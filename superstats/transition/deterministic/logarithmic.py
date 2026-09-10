"""Logarithmic deterministic transition."""

from collections.abc import Sequence
from typing import Any, Dict

import numpy as np

from .deterministic_transition import DeterministicTransition, Prior


class Logarithmic(DeterministicTransition):
    """Deterministic logarithmic transition with an intercept and scale.

    Parameters
    ----------
    bounds          : sequence of two floats or None, optional, default: None
        Lower and upper bounds for the deterministic trajectory. Tuples and
        lists are accepted.
    intercept       : float, Prior, or None, optional, default: None
        Initial value of the trajectory. A `Prior` samples one intercept per
        trajectory; `None` uses the deterministic default prior.
    beta            : float, Prior, or None, optional, default: None
        Logarithmic scale. Positive values produce increasing trajectories
        and negative values produce decreasing trajectories. A `Prior` samples
        one scale per trajectory; `None` uses the deterministic default prior.
    normalize_steps : bool, optional, default: True
        If `True`, use a time axis from 0 to 1. If `False`, use integer
        step indices.

    Notes
    -----
    The trajectory is defined as ``intercept + beta * log1p(t)``. The `sample`
    method returns a dict with keys `deterministic_params`, `hyper_params`,
    and `fixed_params`. Trajectory values are clipped to `bounds`.
    """

    def __init__(
        self,
        bounds: Sequence[float, float] | None = None,
        intercept: float | Prior | None = None,
        beta: float | Prior | None = None,
        normalize_steps: bool = True,
    ):
        super().__init__(bounds=bounds)
        self.normalize_steps = normalize_steps
        self.hyper_specs = {"intercept": intercept, "beta": beta}
        self.transition_name = "logarithmic"

    def sample(self, batch_size: int, num_steps: int) -> Dict[str, Any]:
        """Draw `batch_size` logarithmic trajectories of length `num_steps`.

        Parameters
        ----------
        batch_size : int
            Number of independent trajectories to draw.
        num_steps : int
            Number of time steps per trajectory.

        Returns
        -------
        result : dict
            Dictionary with keys `deterministic_params`, `hyper_params`, and
            `fixed_params`.
        """
        hyper, fixed = self._resolve_hyperparams(batch_size)
        intercept = (
            hyper["intercept"] if "intercept" in hyper else np.full(batch_size, fixed["intercept"], dtype=self.dtype)
        )
        beta = hyper["beta"] if "beta" in hyper else np.full(batch_size, fixed["beta"], dtype=self.dtype)
        index = (
            np.linspace(0.0, 1.0, num_steps, dtype=self.dtype)
            if self.normalize_steps
            else np.arange(num_steps, dtype=self.dtype)
        )
        trajectory = intercept[:, None] + beta[:, None] * np.log1p(index[None, :])

        return {
            "deterministic_params": self._bound(trajectory),
            "hyper_params": hyper,
            "fixed_params": fixed,
        }

    def sample_from_parameters(
        self,
        params: Dict[str, np.ndarray | float],
        batch_size: int,
        num_steps: int,
    ) -> np.ndarray:
        """Generate logarithmic trajectories from resolved parameters.

        Parameters
        ----------
        params : dict
            Resolved `intercept` and `beta` values.
        batch_size : int
            Number of trajectories to generate.
        num_steps : int
            Number of time steps per trajectory.

        Returns
        -------
        trajectory : np.ndarray
            Array of shape `(batch_size, num_steps)` containing the bounded
            trajectories.
        """
        intercept = np.broadcast_to(np.asarray(params["intercept"], dtype=self.dtype), (batch_size,))
        beta = np.broadcast_to(np.asarray(params["beta"], dtype=self.dtype), (batch_size,))
        index = (
            np.linspace(0.0, 1.0, num_steps, dtype=self.dtype)
            if self.normalize_steps
            else np.arange(num_steps, dtype=self.dtype)
        )
        trajectory = intercept[:, None] + beta[:, None] * np.log1p(index[None, :])
        return self._bound(trajectory)
