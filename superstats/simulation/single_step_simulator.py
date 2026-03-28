from typing import Callable
import types
import numpy as np
from numba import njit


class SingleStepSimulator:
    def __init__(
        self,
        model: Callable
    ):
        self.model = self._numbafy(model)
    
    def _numbafy(self, model):
        if not isinstance(model, types.FunctionType) and hasattr(model, 'signatures') and bool(model.signatures):
            try:
                model = njit(model)
            except Exception as e:
                print(f"Could not numbafy the model: {e}")
        return model

    def sample(
        self,
        parameters: list
    ):
        num_steps = len(parameters)
        # Infer output shape
        first = np.asarray(self.model(**parameters[0]))
        sim_data = np.zeros((num_steps,) + first.shape, dtype=first.dtype)
        sim_data[0] = first

        # simulate data
        for i in range(1, num_steps):
            sim_data[i] = self.model(**parameters[i])
        return sim_data
