"""Linear deterministic transition."""

from typing import Dict, Any
import numpy as np

from .deterministic_transition import DeterministicTransition, Prior


class Linear(DeterministicTransition):
    """Deterministic linear transition with an intercept and slope.

    Parameters
    ----------
    intercept       : float, Prior, or None, optional, default: None
        Starting value of the trajectory. A `Prior` samples one intercept
        per trajectory; `None` uses the deterministic default prior.
    slope            : float, Prior, or None, optional, default: None
        Change across the trajectory when `normalize_steps=True`. A `Prior`
        samples one slope per trajectory; `None` uses the deterministic
        default prior.
    normalize_steps : bool, optional, default: True
        If `True`, use a time axis from 0 to 1. If `False`, use integer
        step indices, so the slope is applied at every step.

    Notes
    -----
    The `sample` method returns a dict with keys `deterministic_params`,
    `hyper_params`, and `fixed_params`. Trajectories are returned on the raw coefficient scale.
    """

    def __init__(
        self,
        intercept: float | Prior | None = None,
        slope: float | Prior | None = None,
        normalize_steps: bool = True,
    ):
        super().__init__()

        self.normalize_steps = normalize_steps

        self.hyper_specs = {
            "intercept": intercept,
            "slope": slope,
        }

        self.transition_name = "linear"

    def sample(self, batch_size: int, num_steps: int) -> Dict[str, Any]:
        return self._sample_curve(batch_size, num_steps, Linear.sample_from_parameters)

    def sample_from_parameters(
        self,
        params: Dict[str, np.ndarray | float],
        batch_size: int,
        num_steps: int,
    ) -> np.ndarray:
        intercept = np.broadcast_to(np.asarray(params["intercept"], dtype=self.dtype), (batch_size,))
        slope = np.broadcast_to(np.asarray(params["slope"], dtype=self.dtype), (batch_size,))

        index = (
            np.linspace(0.0, 1.0, num_steps, dtype=self.dtype)
            if self.normalize_steps
            else np.arange(num_steps, dtype=self.dtype)
        )
        trajectory = intercept[:, None] + slope[:, None] * index[None, :]

        return trajectory.astype(self.dtype)
