from typing import Callable, Tuple, Union
import numpy as np
from numba import njit, prange
from superstats.utils.helpers import scaled_sigmoid
from .transition import Transition

@njit
def sample_rw(
    theta: np.ndarray,
    eta: np.ndarray,
    batch_size: int,
    steps: int,
    bounds: np.ndarray,
    transform: bool = False
) -> np.ndarray:
    """
    Generates a single transition from a Gaussian random walk.
    """
    z = np.random.randn(batch_size, steps-1)
    for b in prange(batch_size):
        if transform:
            for t in range(1, steps):
                theta[b, t] = theta[b, t-1] + eta[b] * z[b, t-1]
            theta[b, :] = scaled_sigmoid(
                theta[b, :], bounds[0], bounds[1]
            )
        else:
            theta[b, :] = np.minimum(
                np.maximum(theta[b, :], bounds[0]),
                bounds[1]
            )
    return theta


class RW(Transition):
    def __init__(
        self,
        initial_prior: Callable,
        bounds: Union[Tuple[float, float], np.ndarray],
        sigma: float = 0.1,
        transform: bool = True
    ):
        super().__init__(
            initial_prior=initial_prior,
            bounds=bounds,
            transform=transform
        )
        self.sigma = sigma

    def sample(self, batch_size: int, steps: int) -> np.ndarray:
        theta = np.zeros((batch_size, steps), dtype=np.float32)
        theta[:, 0] = self.initial_prior.sample(batch_size)
        eta = np.abs(np.random.normal(0, self.sigma, size=(batch_size, )))
        return sample_rw(
            theta,
            eta,
            batch_size,
            steps,
            self.bounds,
            self.transform
        )