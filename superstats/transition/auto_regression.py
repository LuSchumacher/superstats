from typing import Tuple
from superstats.transition.auto_regression_drift import AutoRegressionDrift
from superstats.prior import Prior


class AutoRegression(AutoRegressionDrift):
    """Mean-reverting AR(1) process."""

    def __init__(
        self,
        bounds: Tuple[float, float],
        initial_prior: Prior = None,
        phi_prior: Prior | float = None,
        sigma_prior: Prior | float = None,
    ):
        super().__init__(
            bounds=bounds,
            initial_prior=initial_prior,
            phi_prior=phi_prior,
            delta_prior=0.0,
            sigma_prior=sigma_prior
        )