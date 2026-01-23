from typing import Tuple
import numpy as np
from numba import njit
from superstats.utils.helpers import scaled_sigmoid
from .transition import Transition
from prior.prior import Prior

@njit
def sample_rw(
    theta: np.ndarray,
    eta: np.ndarray,
    steps: int,
    bounds: np.ndarray,
    transform: bool = False
) -> np.ndarray:
    """
    Generates a single transition from a Gaussian random walk.
    """
    z = np.random.randn(steps-1)
    if transform:
        for t in range(1, steps):
            theta[t] = theta[t-1] + eta * z[t-1]
        theta[:] = scaled_sigmoid(
            theta, bounds[0], bounds[1]
        )
    else:
        for t in range(1, steps):
            theta[t] = np.minimum(
                np.maximum(theta[t-1] + eta * z[t-1], bounds[0]),
                bounds[1]
            )
    return theta


class RW(Transition):
    def __init__(
        self,
        bounds: Tuple[float, float] | np.ndarray,
        initial_prior: None | Prior = None,
        sigma: float = 0.1,
        transform: bool = False
    ):
        super().__init__(
            initial_prior=initial_prior,
            bounds=bounds,
            transform=transform
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
            self.bounds,
            self.transform
        )
        return {"local_param": theta, "sigma": sigma}