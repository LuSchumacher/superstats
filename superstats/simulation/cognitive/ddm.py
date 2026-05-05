import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def sample_ddm(
    v: np.ndarray,
    a: np.ndarray,
    tau: np.ndarray,
    bias: np.ndarray,
    sigma: float = 1.0,
    dt: float = 0.001,
    max_steps: int = 10000,
) -> np.ndarray:
    """
    Sample from the Diffusion Decision Model (DDM) for decision making.

    This function simulates decision processes using the DDM, where evidence
    accumulates over time with drift rate v, boundary separation a, and noise.
    The simulation stops when a boundary is reached or max_steps is exceeded.

    Parameters
    ----------
    v : np.ndarray
        Drift rates for each trial, shape (num_steps,).
    a : np.ndarray
        Boundary separations for each trial, shape (num_steps,).
    tau : np.ndarray
        Non-decision times for each trial, shape (num_steps,).
    bias : np.ndarray
        Initial biases (as fraction of boundary) for each trial, shape (num_steps,).
    sigma : float, optional
        Diffusion noise standard deviation, default 1.0.
    dt : float, optional
        Time step size, default 0.001.
    max_steps : int, optional
        Maximum number of steps before timeout, default 10000.

    Returns
    -------
    np.ndarray
        Decision data of shape (num_steps, 2), where columns are:
        - Column 0: response time (or -1.0 for timeout)
        - Column 1: choice (1.0 for upper boundary, 0.0 for lower, -1.0 for timeout)
    """
    
    num_steps = v.shape[0]
    data = np.empty((num_steps, 2), dtype=np.float32)
    noise_scale = sigma * np.sqrt(dt)

    for i in prange(num_steps):
        v_t = v[i]
        a_t = a[i]
        t = tau[i]
        x = bias[i] * a_t
        drift_dt = v_t * dt

        for step in range(max_steps):
            t += dt
            x += drift_dt + noise_scale * np.random.normal()
            if x >= a_t:
                data[i, 0] = t
                data[i, 1] = 1.0
                break
            if x <= -a_t:
                data[i, 0] = t
                data[i, 1] = 0.0
                break
        else:
            data[i, 0] = -1.0
            data[i, 1] = -1.0

    return data
