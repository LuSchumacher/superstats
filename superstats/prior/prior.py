import numpy as np
from typing import Literal, Sequence


class Prior:
    """Simple generative prior distribution.

    Parameters
    ----------
    dist  : {"normal", "uniform", "beta", "halfnormal", "dirichlet", "logistic"}
        Distribution type.
    loc   : float, optional, default: 0.0
        Mean for `normal`.
    scale : float, optional, default: 1.0
        Standard deviation for `normal` / `halfnormal`.
    low   : float, optional, default: 0.0
        Lower bound for `uniform`.
    high  : float, optional, default: 1.0
        Upper bound for `uniform`.
    a     : float, optional, default: 1.0
        Alpha (first shape parameter) for `beta`.
    b     : float, optional, default: 1.0
        Beta (second shape parameter) for `beta`.
    alpha : sequence of float or None, optional, default: None
        Concentration parameters for `dirichlet`. Required when
        `dist="dirichlet"` (e.g. [1, 1, 1] for a uniform simplex over
        3 categories).
    """

    def __init__(
        self,
        dist: Literal["normal", "uniform", "beta", "halfnormal", "dirichlet", "logistic"],
        loc: float = 0.0,
        scale: float = 1.0,
        low: float = 0.0,
        high: float = 1.0,
        a: float = 1.0,
        b: float = 1.0,
        alpha: Sequence[float] | None = None,
    ):
        self.dist = dist
        self.loc = loc
        self.scale = scale
        self.low = low
        self.high = high
        self.a = a
        self.b = b
        self.alpha = alpha

    def sample(self, batch_size: int) -> np.ndarray:
        """Draw samples from the prior.

        Parameters
        ----------
        batch_size : int
            Number of samples to draw.

        Returns
        -------
        samples : np.ndarray - shape (batch_size,), or (batch_size, K)
            for `dirichlet` where K is the number of categories

        Raises
        ------
        ValueError
            If `dist="dirichlet"` and `alpha` is None or not a
            vector-like sequence, or if `dist` is not one of the
            supported distributions.
        """
        if self.dist == "normal":
            samples = np.random.normal(self.loc, self.scale, size=batch_size)

        elif self.dist == "halfnormal":
            samples = np.abs(np.random.normal(0.0, self.scale, size=batch_size))

        elif self.dist == "uniform":
            samples = np.random.uniform(self.low, self.high, size=batch_size)

        elif self.dist == "beta":
            samples = np.random.beta(self.a, self.b, size=batch_size)

        elif self.dist == "logistic":
            samples = np.random.logistic(self.loc, self.scale, size=batch_size)

        elif self.dist == "dirichlet":
            if self.alpha is None:
                raise ValueError("alpha must be provided for dirichlet (e.g. [1, 1, 1] for uniform simplex)")

            alpha = np.asarray(self.alpha, dtype=np.float32)

            if alpha.ndim == 0:
                raise ValueError("alpha must be a vector-like sequence")

            samples = np.random.dirichlet(alpha, size=batch_size)

        else:
            raise ValueError(f"Unsupported prior distribution: {self.dist}")

        return samples.astype(np.float32)
