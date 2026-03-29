import numpy as np
from typing import Literal


class Prior:
    """
    Simple generative prior distribution.

    The class wraps a small set of common distributions and allows
    drawing a batch of independent samples for use as shared model
    parameters. Supported distributions are
    ``normal``, ``uniform``, ``beta`` and ``halfnormal``.

    Parameters
    ----------
    dist : {'normal', 'uniform', 'beta', 'halfnormal'}
        Name of the distribution to sample from.
    loc : float, optional
        Mean of the normal distribution.
    scale : float, optional
        Standard deviation for the normal and halfnormal distributions.
    low : float, optional
        Lower bound for the uniform distribution.
    high : float, optional
        Upper bound for the uniform distribution.
    a : float, optional
        ``alpha`` parameter for the beta distribution.
    b : float, optional
        ``beta`` parameter for the beta distribution.
    """

    def __init__(
        self,
        dist: Literal["normal", "uniform", "beta", "halfnormal"],
        loc: float = 0.0,
        scale: float = 1.0,
        low: float = 0.0,
        high: float = 1.0,
        a: float = 1.0,
        b: float = 1.0
    ):
        self.dist = dist
        self.loc = loc
        self.scale = scale
        self.low = low
        self.high = high
        self.a = a
        self.b = b

    def sample(self, batch_size: int) -> np.ndarray:
        """
        Draw a batch of values from the configured distribution.

        Parameters
        ----------
        batch_size : int
            Number of independent samples to generate.

        Returns
        -------
        np.ndarray
            Float32 array of shape ``(batch_size,)`` containing the draws.
        """
        if self.dist == "normal":
            samples = np.random.normal(self.loc, self.scale, size=batch_size)

        elif self.dist == "halfnormal":
            samples = np.abs(np.random.normal(0.0, self.scale, size=batch_size))

        elif self.dist == "uniform":
            samples = np.random.uniform(self.low, self.high, size=batch_size)

        elif self.dist == "beta":
            samples = np.random.beta(self.a, self.b, size=batch_size)

        else:
            raise ValueError(f"Unsupported prior distribution: {self.dist}")

        return samples.astype(np.float32)