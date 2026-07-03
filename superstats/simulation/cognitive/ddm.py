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
    """Sample from the Diffusion Decision Model (DDM) for decision making.

    This function simulates decision processes using the DDM, where evidence
    accumulates over time with drift rate v, boundary separation a, and noise.
    The simulation stops when a boundary is reached or max_steps is exceeded.

    Parameters
    ----------
    v         : np.ndarray of shape (num_trials,)
        Drift rates for each trial.
    a         : np.ndarray of shape (num_trials,)
        Boundary distance from the origin for each trial; decision
        boundaries are at +a and -a.
    tau       : np.ndarray of shape (num_trials,)
        Non-decision times for each trial.
    bias      : np.ndarray of shape (num_trials,)
        Starting point offset, as a fraction of `a` (i.e. the initial
        evidence is `bias * a`). 0 starts at the midpoint between the
        two boundaries; positive values start closer to the upper
        boundary, negative values closer to the lower one.
    sigma     : float, optional, default: 1.0
        Diffusion noise standard deviation.
    dt        : float, optional, default: 0.001
        Time step size.
    max_steps : int, optional, default: 10000
        Maximum number of diffusion steps per trial before timing out.

    Returns
    -------
    data : np.ndarray of shape (num_trials, 2) - decision data, where
        column 0 is the response time (or -1.0 on timeout) and column
        1 is the choice (1.0 for the upper boundary, 0.0 for the
        lower boundary, -1.0 on timeout)
    """

    num_trials = v.shape[0]
    data = np.empty((num_trials, 2), dtype=np.float32)
    noise_scale = sigma * np.sqrt(dt)

    for i in prange(num_trials):
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