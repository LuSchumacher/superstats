from abc import ABC, abstractmethod
from typing import Tuple, Literal
import numpy as np

from superstats.prior import Prior


class Transition(ABC):
    def __init__(
        self,
        bounds: Tuple[float, float] | np.ndarray,
        initial_prior: None | Prior = None
    ):
        if initial_prior is None:
            self.initial_prior = Prior(
                "uniform", low=bounds[0], high=bounds[1]
            )
        else:
            self.initial_prior = initial_prior
        self.bounds = np.asarray(bounds, dtype=np.float32) 
    
    @abstractmethod
    def sample(self, steps: int) -> np.ndarray:
        pass