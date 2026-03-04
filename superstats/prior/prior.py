import numpy as np
from typing import Literal

class Prior:
    def __init__(
        self,
        dist: Literal["normal", "uniform", "beta"],
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

    def sample(self, size: int = 1) -> np.ndarray:
        if self.dist == "normal":
            return np.random.normal(self.loc, self.scale, size)
        elif self.dist == "uniform":
            return np.random.uniform(self.low, self.high, size)
        elif self.dist == "beta":
            return np.random.beta(self.a, self.b, size)
        else:
            raise ValueError(f"Unsupported prior distribution: {self.dist}")