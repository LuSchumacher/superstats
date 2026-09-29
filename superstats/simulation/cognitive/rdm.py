"""Racing Diffusion Model simulator."""

import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def sample_rdm(
    v_base: np.ndarray,
    v_diff: np.ndarray,
    a_base: np.ndarray,
    a_diff: np.ndarray,
    tau: np.ndarray,
    sigma_diff: np.ndarray,
    boost_idx: np.ndarray | None = None,
    bias_idx: np.ndarray | None = None,
    num_accumulators: int = 2,
    sigma_base=1.0,
    dt=0.001,
    max_steps=10000,
):
    """Sample from the Racing Diffusion Model (RDM).

    Simulates independent diffusion accumulators racing from zero toward
    response-specific boundaries. The first accumulator to cross its
    boundary determines the choice and response time.

    Drift rates, boundaries, and noise scales are parameterized on the
    log scale. On each trial, the accumulator selected by `boost_idx`
    receives the positive half-contrast for drift and noise; all other
    accumulators receive the negative half-contrast. The accumulator
    selected by `bias_idx` receives the positive half-contrast for the
    boundary; all other accumulators receive the negative half-contrast.

    Specifically:

        v_boost = exp(v_base + v_diff / 2)
        v_other = exp(v_base - v_diff / 2)

        s_boost = sigma_base * exp(sigma_diff / 2)
        s_other = sigma_base * exp(-sigma_diff / 2)

        a_bias  = exp(a_base + a_diff / 2)
        a_other = exp(a_base - a_diff / 2)

    This contrast coding is unchanged when `num_accumulators > 2`: one
    accumulator receives the positive half-contrast and every remaining
    accumulator receives the negative half-contrast.

    Parameters
    ----------
    v_base           : np.ndarray of shape (num_trials,)
        Trial-wise midpoint of the log drift rates. Its exponential is
        the drift rate when `v_diff` is zero and the geometric mean of
        the boosted and nonboosted drift rates otherwise.
    v_diff           : np.ndarray of shape (num_trials,)
        Trial-wise log drift-rate difference between the boosted and
        nonboosted accumulators.
    a_base           : np.ndarray of shape (num_trials,)
        Trial-wise midpoint of the log boundaries. Its exponential is
        the boundary when `a_diff` is zero and the geometric mean of
        the two boundary levels otherwise.
    a_diff           : np.ndarray of shape (num_trials,)
        Trial-wise log-boundary difference between the accumulator
        selected by `bias_idx` and all other accumulators. Positive
        values give the selected accumulator a higher boundary.
    tau              : np.ndarray of shape (num_trials,)
        Trial-wise nondecision time, added to the simulated decision time.
    sigma_diff       : np.ndarray of shape (num_trials,)
        Trial-wise log-noise difference between the boosted and
        nonboosted accumulators.
    boost_idx        : np.ndarray of shape (num_trials,), optional
        Index of the accumulator receiving the positive drift and noise
        contrasts on each trial. If omitted, accumulator 0 is boosted.
    bias_idx         : np.ndarray of shape (num_trials,), optional
        Index of the accumulator receiving the positive boundary
        contrast on each trial. If omitted, accumulator 0 receives the
        positive boundary contrast.
    num_accumulators : int, default=2
        Number of racing accumulators. Valid accumulator indices range
        from 0 to `num_accumulators - 1`.
    sigma_base       : float, default=1.0
        Baseline noise scale. It is the geometric mean of the boosted
        and nonboosted noise scales.
    dt               : float, default=0.001
        Simulation time-step size in seconds.
    max_steps        : int, default=10000
        Maximum number of diffusion steps per trial.

    Returns
    -------
    dict[str, np.ndarray]
        `"response_time"` contains response times, including
        nondecision time, and `"choice"` contains the winning
        accumulator indices. Both arrays have shape `(num_trials,)`
        and dtype `float32`. Trials without a boundary crossing before
        `max_steps` receive `-1.0` for both values.
    """
    n = v_base.size
    rt = np.full(n, -1.0, np.float32)
    choice = np.full(n, -1.0, np.float32)
    sqrt_dt = np.sqrt(dt)

    for i in prange(n):
        boost = 0 if boost_idx is None else boost_idx[i]
        bias = 0 if bias_idx is None else bias_idx[i]

        dv = v_diff[i] * 0.5
        da = a_diff[i] * 0.5
        ds = sigma_diff[i] * 0.5

        vc = np.exp(v_base[i] + dv) * dt
        vi = np.exp(v_base[i] - dv) * dt
        sc = sigma_base * np.exp(ds) * sqrt_dt
        si = sigma_base * np.exp(-ds) * sqrt_dt
        ar = np.exp(a_base[i] + da)
        ao = np.exp(a_base[i] - da)

        x = np.zeros(num_accumulators, np.float32)

        for step in range(max_steps):
            winner, first = -1, 2.0

            for j in range(num_accumulators):
                old = x[j]

                if j == boost:
                    new = old + vc + sc * np.random.randn()
                else:
                    new = old + vi + si * np.random.randn()

                x[j] = new
                bound = ar if j == bias else ao

                if new >= bound:
                    fraction = (bound - old) / (new - old)

                    if fraction < first:
                        winner, first = j, fraction

            if winner >= 0:
                rt[i] = tau[i] + (step + first) * dt
                choice[i] = winner
                break

    return {"response_time": rt, "choice": choice}
