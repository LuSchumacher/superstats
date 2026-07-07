from typing import Callable
import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def r2_score_per_step(estimates: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Coefficient of determination (R2) between estimates and true values, per step and parameter.

    Parameters
    ----------
    estimates : np.ndarray of shape (num_sim, num_steps, num_params)
        Point estimates (e.g. posterior medians) - compute before
        calling this function.
    targets   : np.ndarray of shape (num_sim, num_steps, num_params)
        Ground-truth parameter values.

    Returns
    -------
    r2 : np.ndarray of shape (num_steps, num_params) - R2 per step
        and parameter
    """
    num_sim, num_steps, num_params = targets.shape
    r2_scores = np.zeros((num_steps, num_params))
    for t in prange(num_steps):
        for p in range(num_params):
            y_true = targets[:, t, p]
            y_pred = estimates[:, t, p]
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2_scores[t, p] = 1.0 - ss_res / (ss_tot + 1e-12)
    return r2_scores


@njit(parallel=True, fastmath=True)
def posterior_contraction_per_step(
    estimates: np.ndarray,
    targets: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Posterior contraction per simulation, step and parameter.

    Parameters
    ----------
    estimates : np.ndarray of shape (num_sim, num_samples, num_steps, num_params)
        Posterior samples.
    targets   : np.ndarray of shape (num_sim, num_steps, num_params)
        Ground-truth parameter values, used to estimate the prior
        variance per step and parameter.
    eps       : float, optional, default: 1e-12
        Numerical floor added to the prior variance denominator.

    Returns
    -------
    contraction : np.ndarray of shape (num_sim, num_steps, num_params)
        - 1 minus the ratio of posterior to prior variance, per
        simulation, step, and parameter
    """
    num_sim, num_steps, num_params = targets.shape
    contraction = np.zeros((num_sim, num_steps, num_params))
    for s in prange(num_sim):
        for t in range(num_steps):
            for p in range(num_params):
                prior_var = np.var(targets[:, t, p])
                denom = max(prior_var, eps)
                post_var = np.var(estimates[s, :, t, p])
                contraction[s, t, p] = 1.0 - post_var / denom
    return contraction


def calibration_error_per_step(
    estimates: np.ndarray,
    targets: np.ndarray,
    resolution: int = 20,
    aggregation: Callable = np.median,
    min_quantile: float = 0.005,
    max_quantile: float = 0.995,
) -> np.ndarray:
    """Marginal calibration error per step and parameter.

    Computes an aggregate score for the marginal calibration error over
    an ensemble of approximate posteriors, per step (time step). The
    calibration error is given as the aggregate (e.g. median) of the
    absolute deviation between an alpha-CI and the relative number of
    inliers from `estimates`, over multiple alphas in (0, 1).

    Parameters
    ----------
    estimates    : np.ndarray of shape (num_sim, num_samples, num_steps, num_params)
        Posterior samples.
    targets      : np.ndarray of shape (num_sim, num_steps, num_params)
        Ground-truth parameter values.
    resolution   : int, optional, default: 20
        Number of credibility intervals (CIs) to consider.
    aggregation  : callable, optional, default: np.median
        Function used to aggregate the per-alpha calibration errors.
        Typically np.mean or np.median.
    min_quantile : float in (0, 1), optional, default: 0.005
        Minimum posterior quantile to consider.
    max_quantile : float in (0, 1), optional, default: 0.995
        Maximum posterior quantile to consider.

    Returns
    -------
    calibration_error : np.ndarray of shape (num_steps, num_params) -
        aggregated calibration error, per step and parameter
    """
    alphas = np.linspace(start=min_quantile, stop=max_quantile, num=resolution)
    regions = 1 - alphas
    lowers = regions / 2
    uppers = 1 - lowers

    quantiles = np.quantile(estimates, [lowers, uppers], axis=1)
    lower_bounds, upper_bounds = quantiles[0], quantiles[1]

    lower_mask = lower_bounds <= targets[None, ...]
    upper_mask = upper_bounds >= targets[None, ...]
    inlier_id = np.logical_and(lower_mask, upper_mask)

    alpha_pred = np.mean(inlier_id, axis=1)
    absolute_errors = np.abs(alpha_pred - alphas[:, None, None])

    return aggregation(absolute_errors, axis=0)


def nrmse_per_step(
    estimates: np.ndarray,
    targets: np.ndarray,
    aggregation: Callable = np.median,
) -> np.ndarray:
    """Per-simulation normalized RMSE between posterior samples and targets, per step.

    RMSE is computed across posterior draws (not aggregated first) for
    each simulation, then normalized by a prior-only bootstrap RMSE
    aggregated across simulations. This follows the "prior"
    normalization scheme: 0 indicates a maximally informative posterior
    (point mass at ground truth), 1 indicates a non-informative
    posterior (equivalent to the prior).

    Parameters
    ----------
    estimates   : np.ndarray of shape (num_sim, num_samples, num_steps, num_params)
        Posterior samples per simulation.
    targets     : np.ndarray of shape (num_sim, num_steps, num_params)
        Target parameter trajectories (themselves prior draws,
        in a simulation-based calibration setting).
    aggregation : callable, optional, default: np.median
        Function used to aggregate the prior-only bootstrap RMSE across
        simulations when computing the normalizer. Typically np.mean
        or np.median.

    Returns
    -------
    nrmse : np.ndarray of shape (num_sim, num_steps, num_params) -
        RMSE across posterior draws, normalized by the aggregated
        prior-only bootstrap RMSE
    """
    num_sim, num_samples, num_steps, num_params = estimates.shape

    err = estimates - targets[:, None, :, :]
    rmse = np.sqrt(np.mean(err**2, axis=1))

    idx = np.random.randint(0, num_sim, size=(num_sim, num_samples))
    prior_bootstrap = targets[idx]
    prior_err = prior_bootstrap - targets[:, None, :, :]
    prior_rmse = np.sqrt(np.mean(prior_err**2, axis=1))
    normalizer = aggregation(prior_rmse, axis=0)

    return rmse / normalizer