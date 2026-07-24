"""Linear deterministic transition."""

from typing import Dict, Any
from collections.abc import Sequence
import numpy as np

from .deterministic_transition import DeterministicTransition, Prior


class Linear(DeterministicTransition):
    """Deterministic linear transition with an intercept and slope.

    Parameters
    ----------
    bounds          : sequence of two floats or None, optional, default: None
        Lower and upper bounds for the deterministic trajectory. Tuples and
        lists are accepted.
    intercept       : float, Prior, or None, optional, default: None
        Starting value of the trajectory. A `Prior` samples one intercept
        per trajectory; `None` uses the deterministic default prior.
    slope           : float, Prior, or None, optional, default: None
        Change across the trajectory when `normalize_steps=True`. A `Prior`
        samples one slope per trajectory; `None` uses the deterministic
        default prior.
    normalize_steps : bool, optional, default: True
        If `True`, use a time axis from 0 to 1. If `False`, use integer
        step indices, so the slope is applied at every step.

    Notes
    -----
    The `sample` method returns a dict with keys `deterministic_params`,
    `hyper_params`, and `fixed_params`. Trajectory values are clipped to `bounds`.
    """

    def __init__(
        self,
        bounds: Sequence[float, float] | None = None,
        intercept: float | Prior | None = None,
        slope: float | Prior | None = None,
        normalize_steps: bool = True,
    ):
        super().__init__(bounds)

        self.normalize_steps = normalize_steps

        self.hyper_specs = {
            "intercept": intercept,
            "slope": slope,
        }

        self.transition_name = "linear"

    def sample(self, batch_size: int, num_steps: int) -> Dict[str, Any]:
        """Draw `batch_size` linear trajectories of length `num_steps`.

        Parameters
        ----------
        batch_size : int
            Number of independent trajectories to draw.
        num_steps  : int
            Number of time steps per trajectory.

        Returns
        -------
        result : dict - dictionary with keys `deterministic_params`,
            `hyper_params`, and `fixed_params`
        """
        hyper, fixed = self._resolve_hyperparams(batch_size)
        intercept = (
            self._sample(hyper["intercept"], batch_size)
            if "intercept" in hyper
            else np.full(batch_size, fixed["intercept"], dtype=self.dtype)
        )
        slope = (
            self._sample(hyper["slope"], batch_size)
            if "slope" in hyper
            else np.full(batch_size, fixed["slope"], dtype=self.dtype)
        )
        if self.normalize_steps:
            index = np.linspace(0.0, 1.0, num_steps, dtype=self.dtype)
        else:
            index = np.arange(num_steps, dtype=self.dtype)
        local = intercept[:, None] + slope[:, None] * index[None, :]

        return {
            "deterministic_params": self._bound(local),
            "hyper_params": hyper,
            "fixed_params": fixed,
        }
