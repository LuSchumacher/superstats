from typing import Tuple
from superstats.transition.auto_regression_drift import AutoRegressionDrift
from superstats.prior import Prior


class RandomWalkDrift(AutoRegressionDrift):
    """Random walk with drift."""

    def __init__(
        self,
        bounds: Tuple[float, float],
        initial_prior: Prior = None,
        delta_prior: Prior | float = None,
        sigma_prior: Prior | float = None,
    ):
        super().__init__(
            bounds=bounds,
            initial_prior=initial_prior,
            phi_prior=1.0,
            delta_prior=delta_prior,
            sigma_prior=sigma_prior
        )