"""Racing Diffusion Model simulator."""

import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def sample_rdm(
    v_base: np.ndarray,
    v_diff: np.ndarray,
    a_base: np.ndarray,
    tau: np.ndarray,
    bias: np.ndarray,
    sigma_diff: np.ndarray,
    num_accumulators: int = 2,
    correct_idx: np.ndarray | None = None,
    sigma_base: float = 1.0,
    dt: float = 0.001,
    max_steps: int = 10000,
) -> np.ndarray:
    """Sample from the Racing Diffusion Model (RDM).

    Simulates `num_accumulators` independent diffusion accumulators racing
    from a starting point of 0 toward their own threshold; the first to
    cross wins and determines the response and response time. On each
    trial, the accumulator at index `correct_idx[i]` is treated as the
    correct/target accumulator: it receives a drift advantage of
    `v_diff`, a bias-scaled threshold, and noise scaled by `sigma_diff`.
    All other accumulators on that trial share the disadvantaged drift,
    an unscaled threshold, and noise fixed at `sigma_base`.

    Parameters
    ----------
    v_base      : np.ndarray of shape (num_trials,)
        Base drift rate shared by all accumulators before the
        correct/incorrect adjustment.
    v_diff      : np.ndarray of shape (num_trials,)
        Drift rate difference between the correct and incorrect
        accumulators. The correct accumulator gets `v_base + v_diff / 2`;
        all other accumulators get `v_base - v_diff / 2`.
    a_base      : np.ndarray of shape (num_trials,)
        Base threshold distance from the origin for each trial.
    tau         : np.ndarray of shape (num_trials,)
        Non-decision times for each trial.
    bias        : np.ndarray of shape (num_trials,)
        Threshold scaling factor in [0, 1] for the correct/target
        accumulator: its threshold is `a_base * bias`. All other
        accumulators use the unscaled threshold `a_base`.
    sigma_diff  : np.ndarray of shape (num_trials,)
        Noise scaling factor in [0, +inf) for the correct/target
        accumulator: its noise SD is `sigma_base * sigma_diff`. All other
        accumulators always use `sigma_base` directly.
    num_accumulators : int
        Number of racing accumulators per trial (fixed across trials).
    correct_idx : np.ndarray of shape (num_trials,), optional
        Index (into `0 .. num_accumulators - 1`) of the correct/target
        accumulator for each trial. If left empty, accumulator 0 is
        treated as correct on every trial.
    sigma_base  : float, optional, default: 1.0
        Diffusion noise standard deviation of the non-correct
        accumulators. Fixed (not estimated per trial) for identifiability.
    dt          : float, optional, default: 0.001
        Time step size.
    max_steps   : int, optional, default: 10000
        Maximum number of diffusion steps per trial before timing out.

    Returns
    -------
    data : np.ndarray of shape (num_trials, 2) - decision data, where
        column 0 is the response time (or -1.0 on timeout) and column
        1 is the index of the winning accumulator (-1.0 on timeout)
    """
    num_trials = v_base.shape[0]
    if correct_idx is None:
        correct_idx = np.zeros(num_trials, dtype=np.float32)
    data = np.empty((num_trials, 2), dtype=np.float32)
    sqrt_dt = np.sqrt(dt)

    for i in prange(num_trials):
        t = tau[i]
        correct = correct_idx[i]

        thresholds = np.empty(num_accumulators, dtype=np.float32)
        drifts = np.empty(num_accumulators, dtype=np.float32)
        noise_scales = np.empty(num_accumulators, dtype=np.float32)
        x = np.zeros(num_accumulators, dtype=np.float32)

        for j in range(num_accumulators):
            if j == correct:
                thresholds[j] = a_base[i] * bias[i]
                drifts[j] = v_base[i] + v_diff[i] / 2
                noise_scales[j] = sigma_base * sigma_diff[i] * sqrt_dt
            else:
                thresholds[j] = a_base[i]
                drifts[j] = v_base[i] - v_diff[i] / 2
                noise_scales[j] = sigma_base * sqrt_dt

        drift_dt = drifts * dt

        winner = -1
        for step in range(max_steps):
            t += dt
            for j in range(num_accumulators):
                x[j] += drift_dt[j] + noise_scales[j] * np.random.normal()
                if x[j] >= thresholds[j]:
                    winner = j
                    break
            if winner >= 0:
                data[i, 0] = t
                data[i, 1] = float(winner)
                break
        else:
            data[i, 0] = -1.0
            data[i, 1] = -1.0

    return data
