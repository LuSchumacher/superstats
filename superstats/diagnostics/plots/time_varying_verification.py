"""Time-varying posterior recovery diagnostics."""

from collections.abc import Mapping, Sequence
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from superstats.diagnostics.metrics import (
    correlation_per_step,
    nrmse_per_step,
    posterior_contraction_per_step,
    calibration_error_per_step,
)

from superstats.utils import prepare_plot_data

from superstats.defaults import (
    BASE_COL_WIDTH,
    BASE_ROW_HEIGHT,
    HSPACE,
    LABEL_FONTSIZE,
    LABEL_PAD,
    METRIC_COLORS,
    TICK_FONTSIZE,
    TITLE_FONTSIZE,
    WSPACE,
    Y_LABEL_PAD,
    METRIC_LABELS,
)


def plot_time_varying_verification(
    estimates: Mapping[str, np.ndarray] | np.ndarray,
    targets: Mapping[str, np.ndarray] | np.ndarray,
    variable_keys: Sequence[str] | None = None,
    variable_names: Sequence[str] | None = None,
    aggregation: Callable = np.median,
    colors: str | Sequence[str] = METRIC_COLORS,
    title_fontsize: int = TITLE_FONTSIZE,
    label_fontsize: int = LABEL_FONTSIZE,
    tick_fontsize: int = TICK_FONTSIZE,
    figsize: tuple[float, float] | None = None,
):
    """Plot recovery diagnostics over steps for time-varying parameters.

    Parameters
    ----------
    estimates      : Mapping[str, np.ndarray] or np.ndarray
        Posterior samples. If a dict, values of shape
        (num_sim, num_samples, num_steps), keyed by variable. If an
        array, shape (num_sim, num_samples, num_steps, num_params)
        directly.
    targets        : Mapping[str, np.ndarray] or np.ndarray
        Ground-truth parameter trajectories, matching the input type
        of `estimates`. If a dict, values of shape
        (num_sim, num_steps). If an array, shape
        (num_sim, num_steps, num_params) directly.
    variable_keys  : sequence of str or None, optional, default: None
        Which variables to select and plot, and in what order, when
        `estimates`/`targets` are dicts. By default, all keys, in
        dict insertion order. Ignored for array input.
    variable_names : sequence of str or None, optional, default: None
        Display names for the plotted columns, in the same order as
        `variable_keys` (or the array's last axis). Defaults to
        `variable_keys` for dict input, or `param_0`, `param_1`, ...
        for array input.
    aggregation    : callable, optional, default: np.median
        Aggregation function passed through to each metric (nrmse,
        contraction, calibration) when collapsing across simulations.
        Typically np.mean or np.median.
    colors         : str or sequence of str, optional, default: METRIC_COLORS
        Row colors, one per metric in the fixed order: correlation,
        nrmse, contraction, calibration. A single str is applied to
        all four rows.
    title_fontsize : int, optional, default: 22
        The font size of the column titles (parameter names). For a single
        parameter, the parameter name is the figure title and metric names
        are the panel titles.
    label_fontsize : int, optional, default: 18
        The font size of the axis label texts and row labels.
    tick_fontsize  : int, optional, default: 16
        The font size of the axis tick labels.
    figsize      : tuple of two floats or None, optional, default: None
        Explicit figure size in inches. If None, the default layout size
        is used.

    Returns
    -------
    fig : plt.Figure - the figure instance for optional saving

    Raises
    ------
    ValueError
        If `estimates`/`targets` are inconsistent (see
        `_prepare_plot_data`), or if `colors` is a sequence whose
        length doesn't match the number of metrics (4).
    """
    if isinstance(colors, str):
        colors = [colors] * 4
    elif len(colors) != 4:
        raise ValueError(f"colors must have 4 entries (one per metric), got {len(colors)}.")

    estimates_arr, targets_arr, param_names = prepare_plot_data(estimates, targets, variable_keys, variable_names)

    num_sim, num_samples, num_steps, num_params = estimates_arr.shape

    metric_values = {
        "correlation": correlation_per_step(estimates_arr, targets_arr, aggregation=aggregation),
        "nrmse": nrmse_per_step(estimates_arr, targets_arr, aggregation=aggregation),
        "contraction": posterior_contraction_per_step(estimates_arr, targets_arr, aggregation=aggregation),
        "calibration": calibration_error_per_step(estimates_arr, targets_arr, aggregation=aggregation),
    }

    metric_keys = ["correlation", "nrmse", "contraction", "calibration"]
    single_parameter = num_params == 1
    if single_parameter:
        num_rows = 2
        num_cols = 2
    else:
        num_rows = len(metric_keys)
        num_cols = num_params
    steps = np.arange(1, num_steps + 1)

    default_figsize = (
        BASE_COL_WIDTH * num_cols,
        BASE_ROW_HEIGHT * num_rows + 0.75 if single_parameter else BASE_ROW_HEIGHT * num_rows,
    )
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=figsize if figsize is not None else default_figsize,
        squeeze=False,
    )

    for row_i, key in enumerate(metric_keys):
        color = colors[row_i]
        values = metric_values[key]

        y_min, y_max = values.min(), values.max()
        pad = (y_max - y_min) * 0.1 or 0.05
        y_lim = (y_min - pad, y_max + pad)

        for param_i in range(num_params):
            if single_parameter:
                plot_row, plot_col = divmod(row_i, 2)
            else:
                plot_row, plot_col = row_i, param_i
            ax = axes[plot_row, plot_col]

            ax.plot(steps, values[:, param_i], color=color, linewidth=2.0)

            ax.set_ylim(y_lim)
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=tick_fontsize)
            ax.set_xlabel("")
            ax.set_ylabel("")

            if single_parameter:
                ax.set_title(
                    METRIC_LABELS[key].replace("\n", " "),
                    fontsize=label_fontsize,
                    pad=10,
                )
                if plot_col == 0:
                    ax.set_ylabel(
                        "Value",
                        fontsize=label_fontsize,
                        labelpad=Y_LABEL_PAD,
                    )
            else:
                if row_i == 0:
                    ax.set_title(param_names[param_i], fontsize=title_fontsize, pad=15)
                if param_i == 0:
                    ax.set_ylabel(
                        METRIC_LABELS[key],
                        fontsize=label_fontsize,
                        labelpad=Y_LABEL_PAD,
                    )
            if plot_row == num_rows - 1:
                ax.set_xlabel("Step", fontsize=label_fontsize, labelpad=LABEL_PAD)

    if single_parameter:
        fig.suptitle(param_names[0], fontsize=title_fontsize, y=0.94)

    if single_parameter:
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.subplots_adjust(hspace=0.75, wspace=WSPACE)
    else:
        fig.tight_layout()
        fig.subplots_adjust(hspace=HSPACE, wspace=WSPACE)
    sns.despine()

    return fig
