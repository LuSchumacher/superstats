"""Diffusion Decision Model simulator."""

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
) -> dict[str, np.ndarray]:
    """Sample from the Diffusion Decision Model (DDM) for decision making.

    This function simulates decision processes using the DDM, where evidence
    accumulates over time with drift rate v, boundary separation a, and noise.
    The simulation stops when a boundary is reached or max_steps is exceeded.

    Parameters
    ----------
    v         : np.ndarray of shape (num_steps,)
        Drift rates for each trial.
    a         : np.ndarray of shape (num_steps,)
        Boundary separation for each trial; decision boundaries are at
        0 (lower) and a (upper).
    tau       : np.ndarray of shape (num_steps,)
        Non-decision times for each trial.
    bias      : np.ndarray of shape (num_steps,)
        Starting point, as a fraction of `a` (i.e. the initial evidence
        is `bias * a`). 0.5 starts at the midpoint between the two
        boundaries; values > 0.5 start closer to the upper boundary,
        values < 0.5 closer to the lower one. Must lie in (0, 1).
    sigma     : float, optional, default: 1.0
        Diffusion noise standard deviation.
    dt        : float, optional, default: 0.001
        Time step size.
    max_steps : int, optional, default: 10000
        Maximum number of diffusion steps per trial before timing out.

    Returns
    -------
    data : dict of np.ndarray
        Named decision data. `"response_time"` contains response times
        (or -1.0 on timeout) and `"choice"` contains choices (1.0 for
        the upper boundary, 0.0 for the lower boundary, -1.0 on timeout).
        Each array has shape (num_steps,).
    """

    num_steps = v.shape[0]
    response_time = np.empty(num_steps, dtype=np.float32)
    choice = np.empty(num_steps, dtype=np.float32)
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
                response_time[i] = t
                choice[i] = 1.0
                break
            if x <= 0.0:
                response_time[i] = t
                choice[i] = 0.0
                break
        else:
            response_time[i] = -1.0
            choice[i] = -1.0

    return {"response_time": response_time, "choice": choice}
