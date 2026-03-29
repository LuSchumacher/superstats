import numpy as np
from numba import njit

@njit
def scaled_sigmoid(
    x: float | np.ndarray,
    lower_bound: float | np.ndarray,
    upper_bound: float | np.ndarray
) -> float | np.ndarray:
    """
    Apply a sigmoid transformation and rescale to a bounded interval.

    This function maps input values to a specified range using a scaled sigmoid.
    The transformation is: lower_bound + (upper_bound - lower_bound) / (1 + exp(-x))

    Parameters
    ----------
    x : float or np.ndarray
        Input value(s) to transform.
    lower_bound : float or np.ndarray
        Lower bound of the output range.
    upper_bound : float or np.ndarray
        Upper bound of the output range.

    Returns
    -------
    float or np.ndarray
        Transformed value(s) in the range [lower_bound, upper_bound].
    """
    return lower_bound + (upper_bound - lower_bound) / (1.0 + np.exp(-x))