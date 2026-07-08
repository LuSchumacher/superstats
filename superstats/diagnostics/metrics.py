from typing import Callable
import numpy as np


def correlation_per_step(
    estimates: np.ndarray,
    targets: np.ndarray,
    aggregation: Callable = np.median,
) -> np.ndarray:
    """Pearson correlation between point estimates and true values, per step and parameter.

    Posterior samples are first collapsed to a point estimate per
    simulation, step, and parameter (using `aggregation`), then the
    Pearson correlation between those point estimates and `targets` is
    computed across simulations.

    Parameters
    ----------
    estimates   : np.ndarray of shape (num_sim, num_samples, num_steps, num_params)
        Posterior samples.
    targets     : np.ndarray of shape (num_sim, num_steps, num_params)
        Ground-truth parameter values.
    aggregation : callable, optional, default: np.median
        Function used to collapse posterior samples into a point
        estimate per simulation, step, and parameter. Typically
        np.mean or np.median.

    Returns
    -------
    correlation : np.ndarray of shape (num_steps, num_params) - Pearson
        correlation per step and parameter
    """
    if estimates.ndim != 4:
        raise ValueError(
            f"estimates must have shape (num_sim, num_samples, num_steps, num_params), got {estimates.shape}."
        )
    if estimates.shape[0] != targets.shape[0] or estimates.shape[2:] != targets.shape[1:]:
        raise ValueError(f"estimates and targets have incompatible shapes: {estimates.shape} vs {targets.shape}.")

    point_est = aggregation(estimates, axis=1)  # (num_sim, num_steps, num_params)

    mean_true = targets.mean(axis=0, keepdims=True)
    mean_pred = point_est.mean(axis=0, keepdims=True)

    cov = np.sum((targets - mean_true) * (point_est - mean_pred), axis=0)
    std_true = np.sqrt(np.sum((targets - mean_true) ** 2, axis=0))
    std_pred = np.sqrt(np.sum((point_est - mean_pred) ** 2, axis=0))
    denom = std_true * std_pred

    return np.where(denom > 1e-12, cov / denom, 0.0)


def posterior_contraction_per_step(
    estimates: np.ndarray,
    targets: np.ndarray,
    aggregation: Callable = np.median,
) -> np.ndarray:
    """Posterior contraction per step and parameter.

    Computes 1 minus the ratio of posterior to prior variance (using
    the unbiased/sample variance, ddof=1) for each simulation, step,
    and parameter, clipped to [0, 1], then aggregates across
    simulations (using `aggregation`) to yield one value per step and
    parameter. Matches the bayesflow `posterior_contraction` metric,
    extended over an additional time-step axis.

    Parameters
    ----------
    estimates   : np.ndarray of shape (num_sim, num_samples, num_steps, num_params)
        Posterior samples.
    targets     : np.ndarray of shape (num_sim, num_steps, num_params)
        Ground-truth parameter values, used to estimate the prior
        variance per step and parameter.
    aggregation : callable, optional, default: np.median
        Function used to aggregate the per-simulation contraction
        values across simulations. Typically np.mean or np.median.

    Returns
    -------
    contraction : np.ndarray of shape (num_steps, num_params) - 1 minus
        the ratio of posterior to prior variance, per step and
        parameter, clipped to [0, 1], aggregated across simulations
    """
    if estimates.ndim != 4:
        raise ValueError(
            f"estimates must have shape (num_sim, num_samples, num_steps, num_params), got {estimates.shape}."
        )
    if estimates.shape[0] != targets.shape[0] or estimates.shape[2:] != targets.shape[1:]:
        raise ValueError(f"estimates and targets have incompatible shapes: {estimates.shape} vs {targets.shape}.")

    post_vars = estimates.var(axis=1, ddof=1)
    prior_vars = targets.var(axis=0, keepdims=True, ddof=1)
    contraction = np.clip(1 - (post_vars / prior_vars), 0, 1)

    return aggregation(contraction, axis=0)


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
    if estimates.ndim != 4:
        raise ValueError(
            f"estimates must have shape (num_sim, num_samples, num_steps, num_params), got {estimates.shape}."
        )
    if estimates.shape[0] != targets.shape[0] or estimates.shape[2:] != targets.shape[1:]:
        raise ValueError(f"estimates and targets have incompatible shapes: {estimates.shape} vs {targets.shape}.")
    if not 0 < min_quantile < max_quantile < 1:
        raise ValueError("Require 0 < min_quantile < max_quantile < 1.")

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
    """Normalized RMSE between posterior samples and targets, per step and parameter.

    RMSE is computed across posterior draws (not aggregated first) for
    each simulation, then normalized by a prior-only bootstrap RMSE
    aggregated across simulations. This follows the "prior"
    normalization scheme: 0 indicates a maximally informative posterior
    (point mass at ground truth), 1 indicates a non-informative
    posterior (equivalent to the prior). The per-simulation ratios are
    then aggregated over simulations (using `aggregation`, mirroring
    the bayesflow convention of aggregating the final metric with the
    same function used for the normalizer), yielding one value per
    step and parameter.

    Parameters
    ----------
    estimates   : np.ndarray of shape (num_sim, num_samples, num_steps, num_params)
        Posterior samples per simulation.
    targets     : np.ndarray of shape (num_sim, num_steps, num_params)
        Target parameter trajectories (themselves prior draws,
        in a simulation-based calibration setting).
    aggregation : callable, optional, default: np.median
        Function used to aggregate both the prior-only bootstrap RMSE
        (for the normalizer) and the final per-simulation nRMSE values
        across simulations. Typically np.mean or np.median.

    Returns
    -------
    nrmse : np.ndarray of shape (num_steps, num_params) - RMSE across
        posterior draws, normalized by the aggregated prior-only
        bootstrap RMSE, aggregated across simulations
    """
    if estimates.ndim != 4:
        raise ValueError(
            f"estimates must have shape (num_sim, num_samples, num_steps, num_params), got {estimates.shape}."
        )
    if estimates.shape[0] != targets.shape[0] or estimates.shape[2:] != targets.shape[1:]:
        raise ValueError(f"estimates and targets have incompatible shapes: {estimates.shape} vs {targets.shape}.")

    num_sim, num_samples, num_steps, num_params = estimates.shape

    err = estimates - targets[:, None, :, :]
    rmse = np.sqrt(np.mean(err**2, axis=1))

    idx = np.random.randint(0, num_sim, size=(num_sim, num_samples))
    prior_bootstrap = targets[idx]
    prior_err = prior_bootstrap - targets[:, None, :, :]
    prior_rmse = np.sqrt(np.mean(prior_err**2, axis=1))
    normalizer = aggregation(prior_rmse, axis=0)

    nrmse = rmse / normalizer

    return aggregation(nrmse, axis=0)
