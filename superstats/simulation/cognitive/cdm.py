"""Circular Diffusion Model simulator."""

import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def sample_cdm(
    v_angle: np.ndarray,
    v_length: np.ndarray,
    a: np.ndarray,
    tau: np.ndarray,
    sigma: float = 1.0,
    dt: float = 0.001,
    max_steps: int = 10000,
) -> dict[str, np.ndarray]:
    """Sample from the Circular Diffusion Model (CDM).

    Simulates a 2D diffusion process starting from the origin, with a
    constant drift specified in polar form, evolving until it crosses a
    circular boundary of radius `a`. The crossing point determines the
    response angle and the number of steps determines the response time.
    On each trial the drift vector has length `v_length` and points in
    direction `v_angle`; the two Cartesian components diffuse independently
    with noise SD `sigma` until the squared radius reaches `a ** 2`.

    Parameters
    ----------
    v_angle   : np.ndarray of shape (num_trials,)
        Direction of the drift vector (in radians) for each trial.
    v_length  : np.ndarray of shape (num_trials,)
        Magnitude of the drift vector for each trial. The Cartesian drift
        components are `v_length * cos(v_angle)` and `v_length * sin(v_angle)`.
    a         : np.ndarray of shape (num_trials,)
        Radius of the circular decision boundary for each trial.
    tau       : np.ndarray of shape (num_trials,)
        Non-decision times for each trial.
    sigma     : float, optional, default: 1.0
        Diffusion noise standard deviation, shared by both Cartesian
        components. Fixed (not estimated per trial) for identifiability,
        since the boundary radius `a` and drift set the overall scale.
    dt        : float, optional, default: 0.001
        Time step size.
    max_steps : int, optional, default: 10000
        Maximum number of diffusion steps per trial before timing out.

    Returns
    -------
    data : dict of np.ndarray
        Named decision data. `"response_time"` contains response times
        (or -5.0 on timeout) and `"choice"` contains response angles in
        radians (or -5.0 on timeout). Each array has shape (num_trials,).
    """
    num_trials = v_angle.shape[0]
    response_time = np.empty(num_trials)
    choice = np.empty(num_trials)

    noise_scale = sigma * np.sqrt(dt)

    mu_cos = np.cos(v_angle)
    mu_sin = np.sin(v_angle)

    for idx in prange(num_trials):
        mu0 = v_length[idx] * mu_cos[idx]
        mu1 = v_length[idx] * mu_sin[idx]

        a_i = a[idx]
        a_sq = a_i * a_i

        x0 = 0.0
        x1 = 0.0

        rt = -5.0
        angle = -5.0

        for i in range(max_steps):
            x0 += mu0 * dt + noise_scale * np.random.randn()
            x1 += mu1 * dt + noise_scale * np.random.randn()

            if x0 * x0 + x1 * x1 >= a_sq:
                rt = tau[idx] + (i + 1) * dt
                angle = np.arctan2(x1, x0)
                break

        response_time[idx] = rt
        choice[idx] = angle

    return {"response_time": response_time, "choice": choice}
