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
    "r2":          "#C1440E",
    "nrmse":       "#E8871A",
    "contraction": "#D4A843",
    "calibration": "#7B3F00",
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
    """
    Summarize (num_sim, num_trials) -> center, lower, upper per trial.

    Parameters
    ----------
    values : np.ndarray, shape (num_sim, num_trials)
    estimator : "median" | "mean"
    uncertainty : "ci" | "std" | "mad"
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


def plot_time_varying_validation(
    true: np.ndarray,
    estimated: np.ndarray,
    param_names: Sequence[str] | None = None,
    estimator: str = "median",
    uncertainty: str = "ci",
    bootstrap_calibration: bool = False,
    n_bootstrap: int = 1000,
    title_fontsize: int = 16,
    label_fontsize: int = 13,
    tick_fontsize: int = 11,
):
    """
    Plot recovery diagnostics over trials for time-varying parameters.

    Parameters
    ----------
    true : np.ndarray, shape (num_sim, num_trials, num_params)
        Ground-truth parameter trajectories.
    estimated : np.ndarray, shape (num_sim, num_trials, num_post_samples, num_params)
        Posterior samples per simulation and trial.
    param_names : list of str, optional
        Column labels. Defaults to param_0, param_1, ...
    estimator : "median" | "mean"
    uncertainty : "ci" | "std" | "mad"
    bootstrap_calibration : bool
        If True, show CI band for calibration error via bootstrap.
    n_bootstrap : int
        Bootstrap samples for calibration CI.
    """
    num_sim, num_trials, num_params = true.shape

    if param_names is None:
        param_names = [f"param_{p}" for p in range(num_params)]

    # -- point estimates for r2 and nrmse --
    point_est = estimated.mean(axis=2)  # (num_sim, num_trials, num_params)

    r2          = r2_score_per_step(true, point_est)
    nrmse       = nrmse_per_step(true, point_est)
    contraction = posterior_contraction_per_step(true, estimated)
    calibration = calibration_error_per_step(
        estimated, true,
        bootstrap=bootstrap_calibration,
        n_bootstrap=n_bootstrap,
    )

    # calibration shape: (num_trials, num_params) or (n_bootstrap, num_trials, num_params)
    calibration_has_ci = bootstrap_calibration

    metrics = {
        "r2":          r2,           # (num_sim, num_trials, num_params)
        "nrmse":       nrmse,        # (num_sim, num_trials, num_params)
        "contraction": contraction,  # (num_sim, num_trials, num_params)
        "calibration": calibration,  # (num_trials, num_params) or (n_bootstrap, ...)
    }

    metric_keys = list(metrics.keys())
    n_rows = len(metric_keys)
    n_cols = num_params
    trials = np.arange(1, num_trials + 1)

    COL_WIDTH, ROW_HEIGHT = 4.0, 2.8
    fig = plt.figure(figsize=(COL_WIDTH * n_cols, ROW_HEIGHT * n_rows))
    gs  = gridspec.GridSpec(n_rows, n_cols, hspace=0.4, wspace=0.3, figure=fig)
    axes = np.array([
        [fig.add_subplot(gs[r, c]) for c in range(n_cols)]
        for r in range(n_rows)
    ])

    for row_i, key in enumerate(metric_keys):
        color = METRIC_COLORS[key]
        data  = metrics[key]

        # collect y range for shared axis
        y_min, y_max = np.inf, -np.inf

        summaries = []
        for p in range(num_params):
            if key == "calibration" and not calibration_has_ci:
                # already aggregated — no band
                center = data[:, p]
                lower  = center
                upper  = center
            else:
                center, lower, upper = _summarize(
                    data[:, :, p], estimator, uncertainty
                )
            summaries.append((center, lower, upper))
            y_min = min(y_min, lower.min())
            y_max = max(y_max, upper.max())

        pad   = (y_max - y_min) * 0.1 or 0.05
        y_lim = (y_min - pad, y_max + pad)

        for col_i, (center, lower, upper) in enumerate(summaries):
            ax = axes[row_i, col_i]

            ax.plot(trials, center, color=color, linewidth=1.8)
            if not np.array_equal(lower, upper):
                ax.fill_between(trials, lower, upper, color=color, alpha=0.25)

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
    plt.tight_layout()
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