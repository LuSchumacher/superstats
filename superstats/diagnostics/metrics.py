import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def r2_score_per_step(true: np.ndarray, estimated: np.ndarray) -> np.ndarray:
    """Coefficient of determination (R2) between true and estimated values, per trial and parameter.

    Parameters
    ----------
    true      : np.ndarray of shape (num_sim, num_trials, num_params)
        Ground-truth parameter values.
    estimated : np.ndarray of shape (num_sim, num_trials, num_params)
        Point estimates (e.g. posterior medians) - compute before
        calling this function.

    Returns
    -------
    r2 : np.ndarray of shape (num_trials, num_params) - R2 per trial
        and parameter
    """
    num_sim, num_trials, num_params = true.shape
    r2_scores = np.zeros((num_trials, num_params))
    for t in prange(num_trials):
        for p in range(num_params):
            y_true = true[:, t, p]
            y_pred = estimated[:, t, p]
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2_scores[t, p] = 1.0 - ss_res / (ss_tot + 1e-12)
    return r2_scores


@njit(parallel=True, fastmath=True)
def nrmse_per_step(true: np.ndarray, estimated: np.ndarray) -> np.ndarray:
    """Normalized RMSE between true and estimated values, per trial and parameter.

    Parameters
    ----------
    true      : np.ndarray of shape (num_sim, num_trials, num_params)
        Ground-truth parameter values.
    estimated : np.ndarray of shape (num_sim, num_trials, num_params)
        Point estimates (e.g. posterior medians).

    Returns
    -------
    nrmse : np.ndarray of shape (num_trials, num_params) - RMSE
        normalized by the true value range, per trial and parameter
    """
    num_sim, num_trials, num_params = true.shape
    nrmse = np.zeros((num_trials, num_params))
    for t in prange(num_trials):
        for p in range(num_params):
            y_true = true[:, t, p]
            y_pred = estimated[:, t, p]
            rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
            true_range = y_true.max() - y_true.min()
            nrmse[t, p] = rmse / (true_range + 1e-12)
    return nrmse


@njit(parallel=True, fastmath=True)
def posterior_contraction_per_step(
    true: np.ndarray,
    estimated: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Posterior contraction per simulation, trial and parameter.

    Parameters
    ----------
    true      : np.ndarray of shape (num_sim, num_trials, num_params)
        Ground-truth parameter values, used to estimate the prior
        variance per trial and parameter.
    estimated : np.ndarray of shape (num_sim, num_trials, num_post_samples, num_params)
        Posterior samples.
    eps       : float, optional, default: 1e-12
        Numerical floor added to the prior variance denominator.

    Returns
    -------
    contraction : np.ndarray of shape (num_sim, num_trials, num_params)
        - 1 minus the ratio of posterior to prior variance, per
        simulation, trial, and parameter
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
    """Per-simulation calibration error, using all simulations for coverage estimation.

    Parameters
    ----------
    estimates  : np.ndarray of shape (num_sim, num_trials, num_post_samples, num_params)
        Posterior samples.
    targets    : np.ndarray of shape (num_sim, num_trials, num_params)
        Ground-truth parameter values.
    resolution : int
        Number of credible interval levels to evaluate.

    Returns
    -------
    calibration_error : np.ndarray of shape (num_sim, num_trials, num_params)
        - median absolute calibration error across credible interval
        levels, per simulation, trial, and parameter
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
    """Calibration error aggregated over simulations.

    Parameters
    ----------
    estimates  : np.ndarray of shape (num_sim, num_trials, num_post_samples, num_params)
        Posterior samples.
    targets    : np.ndarray of shape (num_sim, num_trials, num_params)
        Ground-truth parameter values.
    resolution : int
        Number of credible interval levels to evaluate.

    Returns
    -------
    calibration_error : np.ndarray of shape (num_trials, num_params) -
        median absolute calibration error across credible interval
        levels, per trial and parameter
    """
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
    """Calibration error per trial and parameter.

    Parameters
    ----------
    estimates   : np.ndarray of shape (num_sim, num_trials, num_post_samples, num_params)
        Posterior samples.
    targets     : np.ndarray of shape (num_sim, num_trials, num_params)
        Ground-truth parameter values.
    resolution  : int, optional, default: 20
        Number of credible interval levels.
    bootstrap   : bool, optional, default: False
        If False, returns the error aggregated over simulations. If
        True, returns a bootstrap distribution of the per-simulation
        error resampled over simulations.
    n_bootstrap : int, optional, default: 1000
        Number of bootstrap resamples. Only used when `bootstrap=True`.

    Returns
    -------
    result : np.ndarray - calibration error of shape (num_trials,
        num_params) if `bootstrap=False`, or (n_bootstrap, num_trials,
        num_params) if `bootstrap=True`
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