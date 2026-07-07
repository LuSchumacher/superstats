import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import Sequence

from superstats.diagnostics.metrics import (
    r2_score_per_step,
    nrmse_per_step,
    posterior_contraction_per_step,
    calibration_error_per_step,
)

METRIC_COLORS = {
    "r2":          "#822621",
    "nrmse":       "#C1440E",
    "contraction": "#E8871A",
    "calibration": "#D4A843",
}

METRIC_LABELS = {
    "r2":          "R²",
    "nrmse":       "NRMSE",
    "contraction": "Posterior\nContraction",
    "calibration": "Calibration\nError",
}


def _summarize(
    values: np.ndarray,
    estimator: str,
    uncertainty: str,
):
    """Summarize (num_sim, num_steps) values into center, lower, upper per step.

    Parameters
    ----------
    values      : np.ndarray of shape (num_sim, num_steps)
        Values to summarize across the simulation axis.
    estimator   : {"median", "mean"}
        Center statistic to compute per step.
    uncertainty : {"ci", "std", "mad"}
        Band type to compute around the center per step.

    Returns
    -------
    result : tuple - `(center, lower, upper)`, each an np.ndarray of
        shape (num_steps,)

    Raises
    ------
    ValueError
        If `uncertainty` is not "ci", "std", or "mad".
    """
    if estimator == "median":
        center = np.median(values, axis=0)
    else:
        center = np.mean(values, axis=0)

    if uncertainty == "ci":
        lower = np.percentile(values, 2.5,  axis=0)
        upper = np.percentile(values, 97.5, axis=0)
    elif uncertainty == "std":
        std   = np.std(values, axis=0)
        lower = center - std
        upper = center + std
    elif uncertainty == "mad":
        mad   = np.median(np.abs(values - np.median(values, axis=0)), axis=0)
        lower = center - mad
        upper = center + mad
    else:
        raise ValueError(f"Unknown uncertainty: '{uncertainty}'")

    return center, lower, upper


def plot_time_varying_verification(
    estimates: np.ndarray,
    targets: np.ndarray,
    param_names: Sequence[str] | None = None,
    estimator: str = "median",
    uncertainty: str = "ci",
    title_fontsize: int = 16,
    label_fontsize: int = 13,
    tick_fontsize: int = 11,
):
    """Plot recovery diagnostics over steps for time-varying parameters.

    Parameters
    ----------
    estimates       : np.ndarray of shape (num_sim, num_samples, num_steps, num_params)
        Posterior samples per simulation and step.
    targets         : np.ndarray of shape (num_sim, num_steps, num_params)
        Ground-truth parameter trajectories.
    param_names     : list of str or None, optional, default: None
        Column labels. Defaults to `param_0`, `param_1`, ... when not
        supplied.
    estimator       : {"median", "mean"}, optional, default: "median"
        Used for the nrmse and contraction center lines (which retain
        the simulation axis).
    uncertainty     : {"ci", "std", "mad"}, optional, default: "ci"
        Used for the nrmse and contraction uncertainty bands.
    title_fontsize  : int, optional, default: 16
        The font size of the column titles (parameter names).
    label_fontsize  : int, optional, default: 13
        The font size of the axis label texts and row labels.
    tick_fontsize   : int, optional, default: 11
        The font size of the axis tick labels.

    Returns
    -------
    fig : plt.Figure - the figure instance for optional saving
    """
    num_sim, num_samples, num_steps, num_params = estimates.shape

    if param_names is None:
        param_names = [f"param_{p}" for p in range(num_params)]

    # -- point estimates: posterior median per sim per step --
    point_est = np.median(estimates, axis=1)  # (num_sim, num_steps, num_params)

    # r2: already aggregated across sims -> (num_steps, num_params)
    r2 = r2_score_per_step(point_est, targets)

    # nrmse: per sim -> (num_sim, num_steps, num_params)
    nrmse = nrmse_per_step(estimates, targets, aggregation=np.median if estimator == "median" else np.mean)

    # contraction: per sim -> (num_sim, num_steps, num_params)
    contraction = posterior_contraction_per_step(estimates, targets)

    # calibration: aggregated across sims -> (num_steps, num_params)
    calibration = calibration_error_per_step(
        estimates, targets,
        aggregation=np.median if estimator == "median" else np.mean,
    )

    # metrics whose values are mathematically bounded to [0, 1]; the CI/std/mad
    # band is computed arithmetically and can otherwise overshoot these bounds
    # even though no individual sample violates them
    BOUNDED_UNIT_INTERVAL = {"nrmse", "contraction"}

    metric_keys = ["r2", "nrmse", "contraction", "calibration"]
    n_rows = len(metric_keys)
    n_cols = num_params
    steps = np.arange(1, num_steps + 1)

    COL_WIDTH, ROW_HEIGHT = 4.0, 2.8
    fig = plt.figure(figsize=(COL_WIDTH * n_cols, ROW_HEIGHT * n_rows))
    gs  = gridspec.GridSpec(n_rows, n_cols, hspace=0.4, wspace=0.3, figure=fig)
    axes = np.array([
        [fig.add_subplot(gs[r, c]) for c in range(n_cols)]
        for r in range(n_rows)
    ])

    for row_i, key in enumerate(metric_keys):
        color = METRIC_COLORS[key]

        y_min, y_max = np.inf, -np.inf
        summaries = []

        for p in range(num_params):
            if key == "r2":
                # (num_steps,) — no CI band
                center = r2[:, p]
                lower  = center
                upper  = center

            elif key == "nrmse":
                # (num_sim, num_steps) — has CI band
                center, lower, upper = _summarize(
                    nrmse[:, :, p], estimator, uncertainty
                )

            elif key == "contraction":
                # (num_sim, num_steps) — has CI band
                center, lower, upper = _summarize(
                    contraction[:, :, p], estimator, uncertainty
                )

            elif key == "calibration":
                # (num_steps,) — no CI band
                center = calibration[:, p]
                lower  = center
                upper  = center

            if key in BOUNDED_UNIT_INTERVAL:
                center = np.clip(center, 0.0, 1.0)
                lower  = np.clip(lower, 0.0, 1.0)
                upper  = np.clip(upper, 0.0, 1.0)

            summaries.append((center, lower, upper))
            y_min = min(y_min, lower.min())
            y_max = max(y_max, upper.max())

        pad   = (y_max - y_min) * 0.1 or 0.05
        y_lim = (y_min - pad, y_max + pad)

        for col_i, (center, lower, upper) in enumerate(summaries):
            ax = axes[row_i, col_i]

            ax.plot(steps, center, color=color, linewidth=1.8)
            if not np.array_equal(lower, upper):
                ax.fill_between(steps, lower, upper, color=color, alpha=0.25, edgecolor="none")

            ax.set_ylim(y_lim)
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=tick_fontsize)
            ax.set_xlabel("")
            ax.set_ylabel("")

            if row_i == 0:
                ax.set_title(param_names[col_i], fontsize=title_fontsize)
            if row_i == n_rows - 1:
                ax.set_xlabel("Step", fontsize=label_fontsize)

    # -- row labels --
    plt.draw()

    for row_i, key in enumerate(metric_keys):
        ax0  = axes[row_i, 0]
        bbox = ax0.get_position()
        fig.text(
            0.01,
            bbox.y0 + bbox.height / 2,
            METRIC_LABELS[key],
            ha="center", va="center",
            fontsize=label_fontsize,
            rotation=90,
        )

    fig.subplots_adjust(left=0.1)
    sns.despine()

    return fig