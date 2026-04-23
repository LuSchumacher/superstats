import numpy as np
from typing import Literal


class Prior:
    """
    Simple generative prior distribution.

    The class wraps a small set of common distributions and allows
    drawing a batch of independent samples for use as shared model
    parameters. Supported distributions are
    ``normal``, ``uniform``, ``beta`` and ``halfnormal``.

    Attributes
    ----------
    dist : str
        Name of the distribution ('normal', 'uniform', 'beta', 'halfnormal').
    loc : float
        Mean of the normal distribution.
    scale : float
        Standard deviation for normal and halfnormal distributions.
    low : float
        Lower bound for the uniform distribution.
    high : float
        Upper bound for the uniform distribution.
    a : float
        Alpha parameter for the beta distribution.
    b : float
        Beta parameter for the beta distribution.
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
        """
        Initialize a prior distribution.

        Parameters
        ----------
        dist : {'normal', 'uniform', 'beta', 'halfnormal'}
            Name of the distribution to sample from.
        loc : float, optional
            Mean of the normal distribution (default: 0.0).
        scale : float, optional
            Standard deviation for normal and halfnormal (default: 1.0).
        low : float, optional
            Lower bound for uniform distribution (default: 0.0).
        high : float, optional
            Upper bound for uniform distribution (default: 1.0).
        a : float, optional
            Alpha parameter for beta distribution (default: 1.0).
        b : float, optional
            Beta parameter for beta distribution (default: 1.0).
        """
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