from typing import Tuple, Literal
import numpy as np
from numba import njit

from .transition import Transition
from superstats.utils.helpers import scaled_sigmoid
from superstats.prior import Prior


@njit
def sample_rw(
    theta: np.ndarray,
    eta: np.ndarray,
    steps: int,
    bounds: np.ndarray
) -> np.ndarray:
    """
    Generates a single transition from a Gaussian random walk.
    """
    z = np.random.randn(steps-1)
    for t in range(1, steps):
        theta[t] = theta[t-1] + eta * z[t-1]
    theta[:] = scaled_sigmoid(
        theta, bounds[0], bounds[1]
    )
    return theta


class RandomWalk(Transition):
    def __init__(
        self,
        bounds: Tuple[float, float] | np.ndarray,
        initial_prior: None | Prior = None,
        sigma: float = 0.1
    ):
        super().__init__(
            initial_prior=initial_prior,
            bounds=bounds
        )
        self.sigma = sigma

    def sample(self, steps: int) -> dict:
        theta = np.zeros((steps), dtype=np.float32)
        theta[0] = self.initial_prior.sample()
        sigma = np.abs(np.random.normal(0.0, self.sigma))
        theta[:] = sample_rw(
            theta,
            sigma,
            steps,
            self.bounds
        )
        return {"local_param": theta, "sigma": sigma}