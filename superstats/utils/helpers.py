import numpy as np
from numba import njit

@njit
def scaled_sigmoid(
    x: float | np.ndarray,
    lower_bound: float | np.ndarray,
    upper_bound: float | np.ndarray
) -> float:
    """
    Apply a sigmoid transformation and rescale to a bounded interval.
    """
    return lower_bound + (upper_bound - lower_bound) / (1.0 + np.exp(-x))