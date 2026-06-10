import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def r2_score_per_step(true: np.ndarray, estimated: np.ndarray) -> np.ndarray:
    """
    R² score per simulation, trial and parameter.

    Parameters
    ----------
    true : np.ndarray, shape (num_sim, num_trials, num_params)
    estimated : np.ndarray, shape (num_sim, num_trials, num_params)

    Returns
    -------
    np.ndarray, shape (num_sim, num_trials, num_params)
    """
    num_sim, num_trials, num_params = true.shape
    r2_scores = np.zeros((num_sim, num_trials, num_params))
    for s in prange(num_sim):
        for t in range(num_trials):
            for p in range(num_params):
                y_true = true[s, :t+1, p]
                y_pred = estimated[s, :t+1, p]
                ss_res = np.sum((y_true - y_pred) ** 2)
                ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                r2_scores[s, t, p] = 1.0 - ss_res / (ss_tot + 1e-12)
    return r2_scores


@njit(parallel=True, fastmath=True)
def nrmse_per_step(true: np.ndarray, estimated: np.ndarray) -> np.ndarray:
    """
    Normalised RMSE per simulation, trial and parameter.

    Parameters
    ----------
    true : np.ndarray, shape (num_sim, num_trials, num_params)
    estimated : np.ndarray, shape (num_sim, num_trials, num_params)

    Returns
    -------
    np.ndarray, shape (num_sim, num_trials, num_params)
    """
    num_sim, num_trials, num_params = true.shape
    nrmse = np.zeros((num_sim, num_trials, num_params))
    for s in prange(num_sim):
        for t in range(num_trials):
            for p in range(num_params):
                y_true = true[s, :t+1, p]
                y_pred = estimated[s, :t+1, p]
                rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
                true_range = y_true.max() - y_true.min()
                nrmse[s, t, p] = rmse / (true_range + 1e-12)
    return nrmse


@njit(parallel=True, fastmath=True)
def posterior_contraction_per_step(
    true: np.ndarray,
    estimated: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Posterior contraction per simulation, trial and parameter.

    Parameters
    ----------
    true : np.ndarray, shape (num_sim, num_trials, num_params)
    estimated : np.ndarray, shape (num_sim, num_trials, num_post_samples, num_params)
    eps : float

    Returns
    -------
    np.ndarray, shape (num_sim, num_trials, num_params)
    """
    num_sim, num_trials, num_params = true.shape
    contraction = np.zeros((num_sim, num_trials, num_params))
    for s in prange(num_sim):
        for t in range(num_trials):
            for p in range(num_params):
                prior_var = np.var(true[:, t, p])
                denom = max(prior_var, eps)
                post_var = np.var(estimated[s, t, :, p])
                contraction[s, t, p] = 1.0 - post_var / denom
    return contraction


@njit(parallel=True, fastmath=True)
def _calibration_error_per_step_core(
    estimates: np.ndarray,
    targets: np.ndarray,
    resolution: int,
) -> np.ndarray:
    """
    Core calibration error computation returning (num_sim, num_trials, num_params).
    Each sim's calibration is computed using all sims for coverage estimation.
    """
    num_sim, num_trials, num_post_samples, num_params = estimates.shape
    calibration_error = np.zeros((num_sim, num_trials, num_params))

    for s in prange(num_sim):
        for t in range(num_trials):
            for p in range(num_params):
                errors = np.zeros(resolution)
                for i in range(resolution):
                    alpha   = 0.05 + i * (0.90 / (resolution - 1))
                    lower_q = (1.0 - alpha) / 2.0
                    upper_q = 1.0 - lower_q
                    in_interval = 0.0
                    for s2 in range(num_sim):
                        samps = estimates[s2, t, :, p]
                        lo = np.quantile(samps, lower_q)
                        hi = np.quantile(samps, upper_q)
                        if targets[s2, t, p] >= lo and targets[s2, t, p] <= hi:
                            in_interval += 1.0
                    coverage = in_interval / num_sim
                    errors[i] = abs(coverage - alpha)
                calibration_error[s, t, p] = np.median(errors)

    return calibration_error


@njit(parallel=True, fastmath=True)
def _calibration_error_aggregated(
    estimates: np.ndarray,
    targets: np.ndarray,
    resolution: int,
) -> np.ndarray:
    """Aggregated calibration error returning (num_trials, num_params)."""
    num_sim, num_trials, num_post_samples, num_params = estimates.shape
    calibration_error = np.zeros((num_trials, num_params))

    for t in prange(num_trials):
        for p in range(num_params):
            errors = np.zeros(resolution)
            for i in range(resolution):
                alpha   = 0.05 + i * (0.90 / (resolution - 1))
                lower_q = (1.0 - alpha) / 2.0
                upper_q = 1.0 - lower_q
                in_interval = 0.0
                for s in range(num_sim):
                    samps = estimates[s, t, :, p]
                    lo = np.quantile(samps, lower_q)
                    hi = np.quantile(samps, upper_q)
                    if targets[s, t, p] >= lo and targets[s, t, p] <= hi:
                        in_interval += 1.0
                coverage = in_interval / num_sim
                errors[i] = abs(coverage - alpha)
            calibration_error[t, p] = np.median(errors)

    return calibration_error


def calibration_error_per_step(
    estimates: np.ndarray,
    targets: np.ndarray,
    resolution: int = 20,
    bootstrap: bool = False,
    n_bootstrap: int = 1000,
) -> np.ndarray:
    """
    Calibration error per trial and parameter.

    Parameters
    ----------
    estimates : np.ndarray, shape (num_sim, num_trials, num_post_samples, num_params)
    targets   : np.ndarray, shape (num_sim, num_trials, num_params)
    resolution : int
        Number of credible interval levels.
    bootstrap : bool
        If False (default), returns aggregated (num_trials, num_params).
        If True, returns bootstrap distribution (n_bootstrap, num_trials, num_params).
    n_bootstrap : int
        Number of bootstrap samples. Only used when bootstrap=True.

    Returns
    -------
    np.ndarray
        Shape (num_trials, num_params) if bootstrap=False,
        (n_bootstrap, num_trials, num_params) if bootstrap=True.
    """
    estimates = np.ascontiguousarray(estimates, dtype=np.float64)
    targets   = np.ascontiguousarray(targets,   dtype=np.float64)

    if not bootstrap:
        return _calibration_error_aggregated(estimates, targets, resolution)

    num_sim = estimates.shape[0]
    per_sim = _calibration_error_per_step_core(estimates, targets, resolution)

    num_trials, num_params = per_sim.shape[1], per_sim.shape[2]
    boot = np.zeros((n_bootstrap, num_trials, num_params))
    for b in range(n_bootstrap):
        idx = np.random.randint(0, num_sim, size=num_sim)
        boot[b] = per_sim[idx].mean(axis=0)

    return boot