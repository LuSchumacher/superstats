import numpy as np
from typing import Literal, Optional, Sequence


class Prior:
    """
    Simple generative prior distribution.

    Supported distributions:
    - normal
    - uniform
    - beta
    - halfnormal
    - dirichlet
    """

    def __init__(
        self,
        dist: Literal["normal", "uniform", "beta", "halfnormal", "dirichlet"],
        loc: float = 0.0,
        scale: float = 1.0,
        low: float = 0.0,
        high: float = 1.0,
        a: float = 1.0,
        b: float = 1.0,
        alpha: Sequence[float] = None,
    ):
        """
        Parameters
        ----------
        dist : str
            Distribution type.
        loc : float
            Mean for normal.
        scale : float
            Std for normal/halfnormal.
        low : float
            Lower bound for uniform.
        high : float
            Upper bound for uniform.
        a : float
            Alpha for beta.
        b : float
            Beta for beta.
        alpha : sequence of float, optional
            Dirichlet concentration parameters.
        """
        self.dist = dist
        self.loc = loc
        self.scale = scale
        self.low = low
        self.high = high
        self.a = a
        self.b = b
        self.alpha = alpha

    def sample(self, batch_size: int) -> np.ndarray:
        """
        Draw samples from the prior.

        Returns
        -------
        np.ndarray
            Shape (batch_size,) or (batch_size, K) for Dirichlet.
        """

        if self.dist == "normal":
            samples = np.random.normal(self.loc, self.scale, size=batch_size)

        elif self.dist == "halfnormal":
            samples = np.abs(
                np.random.normal(0.0, self.scale, size=batch_size)
            )

        elif self.dist == "uniform":
            samples = np.random.uniform(self.low, self.high, size=batch_size)

        elif self.dist == "beta":
            samples = np.random.beta(self.a, self.b, size=batch_size)

        elif self.dist == "dirichlet":
            if self.alpha is None:
                raise ValueError(
                    "alpha must be provided for dirichlet "
                    "(e.g. [1, 1, 1] for uniform simplex)"
                )

            alpha = np.asarray(self.alpha, dtype=np.float32)

            if alpha.ndim == 0:
                raise ValueError("alpha must be a vector-like sequence")

            samples = np.random.dirichlet(alpha, size=batch_size)

        else:
            raise ValueError(f"Unsupported prior distribution: {self.dist}")

        return samples.astype(np.float32)