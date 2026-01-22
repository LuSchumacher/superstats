from abc import ABC, abstractmethod
from typing import Callable, Tuple, Union
import numpy as np

class Transition(ABC):
    def __init__(
        self,
        initial_prior: Callable,
        bounds: Union[Tuple[float, float], np.ndarray],
        transform: bool = True
    ):
        self.initial_prior = initial_prior
        self.bounds = np.asarray(bounds, dtype=np.float32)
        self.transform = transform
    
    @abstractmethod
    def sample(self, batch_size: int, steps: int) -> np.ndarray:
        pass