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


# import logging
# from collections.abc import Callable, Mapping, Sequence

# import numpy as np


# def root_mean_squared_error(
#     estimates: Mapping[str, np.ndarray] | np.ndarray,
#     targets: Mapping[str, np.ndarray] | np.ndarray,
#     variable_keys: Sequence[str] | None = None,
#     variable_names: Sequence[str] | None = None,
#     test_quantities: dict[str, Callable] | None = None,
#     normalize: str | None = "prior",
#     aggregation: Callable = np.median,
# ) -> np.ndarray:
#     """Compute the (Normalized) Root Mean Squared Error (RMSE/NRMSE) per step, for the given posterior and prior samples.

#     Datasets and posterior draws are aggregated away, but the `num_steps`
#     axis is kept in the output, so each step gets its own (N)RMSE, the
#     same way `nrmse_per_step` keeps a `num_trials` axis instead of
#     collapsing it.

#     The values of the default normalization (`prior`) should be
#     interpreted as 0 indicating the most informative (posterior is a
#     point mass at ground truth) and 1 indicating non-informative
#     (posterior equals prior) results.

#     Parameters
#     ----------
#     estimates       : np.ndarray or dict[str, np.ndarray]
#         Posterior samples, either as a NumPy array of shape
#         (num_datasets, num_draws_post, num_steps, num_variables) or as
#         a dictionary mapping variable names to arrays of that shape.
#         Comprises `num_draws_post` random draws from the posterior
#         distribution for each dataset and step.
#     targets          : np.ndarray or dict[str, np.ndarray]
#         Ground-truth parameter trajectories, either as a NumPy array
#         of shape (num_datasets, num_steps, num_variables) or as a
#         dictionary mapping variable names to arrays of that shape.
#         Comprises `num_datasets` ground truths (themselves prior
#         draws, in a simulation-based calibration setting).
#     variable_keys    : sequence of str or None, optional, default: None
#         Select keys from the dictionaries provided in `estimates` and
#         `targets`. By default, selects all keys.
#     variable_names   : sequence of str or None, optional, default: None
#         Optional variable names to select, in the same order as
#         `variable_keys`.
#     test_quantities  : dict of str to callable or None, optional, default: None
#         A dict that maps plot titles to functions that compute test
#         quantities based on estimate/target draws.
#         The dict keys are automatically added to `variable_keys` and
#         `variable_names`. Test quantity functions are expected to
#         accept a dict of draws with shape `(batch_size, ...)` as the
#         first (typically only) positional argument and return a NumPy
#         array of shape `(batch_size,)`. The functions do not have to
#         deal with an additional sample dimension, as appropriate
#         reshaping is done internally.
#     normalize        : {"mean", "range", "median", "iqr", "std", "prior"} or None, optional, default: "prior"
#         Whether and how to normalize the RMSE using statistics of the
#         prior (`targets`) samples, computed independently per step.
#         `False` is also accepted as an alias for None (no normalization).
#     aggregation      : callable, optional, default: np.median
#         Function to aggregate the RMSE across datasets (and, for
#         `normalize="prior"`, across bootstrap draws). Typically
#         `np.mean` or `np.median`.

#     Returns
#     -------
#     nrmse : np.ndarray of shape (num_steps, num_variables) - the
#         aggregated (N)RMSE per step and variable

#     Raises
#     ------
#     ValueError
#         If `normalize` is not one of the recognized modes.

#     Notes
#     -----
#     Aggregation is performed after computing the RMSE for each
#     posterior draw, instead of first aggregating the posterior draws
#     and then computing the RMSE between aggregates and ground truths.
#     """

#     if normalize:
#         logging.warning(
#             "Using new default normalize='prior' for a more dynamic range. "
#             "To reproduce previous behavior, set normalize='range'."
#         )

#     # Optionally, compute and prepend test quantities from draws
#     if test_quantities is not None:
#         updated_data = compute_test_quantities(
#             targets=targets,
#             estimates=estimates,
#             variable_keys=variable_keys,
#             variable_names=variable_names,
#             test_quantities=test_quantities,
#         )
#         variable_names = updated_data["variable_names"]
#         variable_keys = updated_data["variable_keys"]
#         estimates = updated_data["estimates"]
#         targets = updated_data["targets"]

#     samples = dicts_to_arrays(
#         estimates=estimates,
#         targets=targets,
#         variable_keys=variable_keys,
#         variable_names=variable_names,
#     )

#     # estimates: (num_datasets, num_draws_post, num_steps, num_variables)
#     # targets:   (num_datasets, num_steps, num_variables)
#     # insert the draws axis into targets so it broadcasts against estimates,
#     # without collapsing num_steps
#     err = samples["estimates"] - samples["targets"][:, None, :, :]
#     rmse = np.sqrt(np.mean(err**2, axis=1))  # -> (num_datasets, num_steps, num_variables)

#     targets = samples["targets"]

#     match normalize:
#         case None | False:
#             normalizer = np.array(1.0)

#         case "mean":
#             normalizer = np.mean(targets, axis=0)

#         case "median":
#             normalizer = np.median(targets, axis=0)

#         case "range":
#             normalizer = targets.max(axis=0) - targets.min(axis=0)

#         case "std":
#             normalizer = np.std(targets, axis=0, ddof=0)

#         case "iqr":
#             q75 = np.percentile(targets, 75, axis=0)
#             q25 = np.percentile(targets, 25, axis=0)
#             normalizer = q75 - q25

#         case "prior":
#             num_datasets, num_draws = samples["estimates"].shape[:2]

#             # bootstrap prior-only predictions from empirical prior samples in targets
#             idx = np.random.randint(0, num_datasets, size=(num_datasets, num_draws))
#             prior_bootstrap = targets[idx]  # -> (num_datasets, num_draws, num_steps, num_variables)

#             prior_err = prior_bootstrap - targets[:, None, :, :]
#             prior_rmse = np.sqrt(np.mean(prior_err**2, axis=1))  # -> (num_datasets, num_steps, num_variables)
#             normalizer = aggregation(prior_rmse, axis=0)  # -> (num_steps, num_variables)

#         case _:
#             raise ValueError(f"Unknown normalization mode: {normalize}")

#     rmse = rmse / normalizer
#     rmse = aggregation(rmse, axis=0)  # aggregate over datasets -> (num_steps, num_variables)

#     return rmse