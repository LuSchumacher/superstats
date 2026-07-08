from collections.abc import Mapping, Sequence

import numpy as np
import bayesflow as bf

from superstats.utils import prepare_plot_data

from superstats.defaults import (
    BASE_COLOR,
)

import matplotlib.pyplot as plt

plt.rcParams["axes.axisbelow"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Palatino", "Palatino Linotype", "DejaVu Serif"]


def plot_recovery(
    estimates: Mapping[str, np.ndarray] | np.ndarray,
    targets: Mapping[str, np.ndarray] | np.ndarray,
    variable_keys: Sequence[str] | None = None,
    variable_names: Sequence[str] | None = None,
    color: str = BASE_COLOR,
    title_fontsize: int = 22,
    label_fontsize: int = 18,
    metric_fontsize: int = 18,
    tick_fontsize: int = 16,
    **kwargs,
):
    """Plot time-invariant parameter recovery.

    Thin wrapper around `bf.diagnostics.plots.recovery` that accepts
    dict-or-array input via `prepare_plot_data`, consistent with
    `plot_time_varying_verification`.

    Parameters
    ----------
    estimates      : Mapping[str, np.ndarray] or np.ndarray
        Posterior estimates. If a dict, per-key arrays sharing the
        same leading shape, keyed by variable. If an array, shape
        (num_sims, num_samples, num_params) directly.
    targets        : Mapping[str, np.ndarray] or np.ndarray
        Ground-truth values, matching the input type of `estimates`.
        If a dict, per-key arrays. If an array, shape
        (num_sims, num_params) directly.
    variable_keys  : sequence of str or None, optional, default: None
        Which keys to select and plot, and in what order, when
        `estimates`/`targets` are dicts. By default, all keys, in
        dict insertion order. Ignored for array input.
    variable_names : sequence of str or None, optional, default: None
        Display names for the plotted columns. Defaults to
        `variable_keys` (dict input) or `param_0`, `param_1`, ...
        (array input).
    color          : str, optional, default: "#822621"
        Base color for plotted lines and fills.
    title_fontsize : int, optional, default: 22
        The font size of the panel titles.
    label_fontsize : int, optional, default: 18
        The font size of the axis label texts.
    metric_fontsize : int, optional, default: 18
        The font size of the displayed recovery metric text.
    tick_fontsize  : int, optional, default: 16
        The font size of the axis tick labels.
    **kwargs
        Forwarded to `bf.diagnostics.plots.recovery` (e.g. `figsize`,
        `num_row`, `num_col`).

    Returns
    -------
    fig : plt.Figure - the recovery diagnostic figure
    """
    estimates_arr, targets_arr, names = prepare_plot_data(estimates, targets, variable_keys, variable_names)

    return bf.diagnostics.plots.recovery(
        estimates=estimates_arr,
        targets=targets_arr,
        variable_names=names,
        color=color,
        title_fontsize=title_fontsize,
        label_fontsize=label_fontsize,
        metric_fontsize=metric_fontsize,
        tick_fontsize=tick_fontsize,
        **kwargs,
    )


def plot_calibration(
    estimates: Mapping[str, np.ndarray] | np.ndarray,
    targets: Mapping[str, np.ndarray] | np.ndarray,
    variable_keys: Sequence[str] | None = None,
    variable_names: Sequence[str] | None = None,
    color: str = BASE_COLOR,
    title_fontsize: int = 22,
    label_fontsize: int = 18,
    metric_fontsize: int = 18,
    tick_fontsize: int = 16,
    **kwargs,
):
    """Plot time-invariant calibration (ECDF).

    Thin wrapper around `bf.diagnostics.plots.calibration_ecdf` that
    accepts dict-or-array input via `prepare_plot_data`, consistent
    with `plot_time_varying_verification`.

    Parameters
    ----------
    estimates      : Mapping[str, np.ndarray] or np.ndarray
        Posterior estimates. If a dict, per-key arrays sharing the
        same leading shape, keyed by variable. If an array, shape
        (num_sims, num_samples, num_params) directly.
    targets        : Mapping[str, np.ndarray] or np.ndarray
        Ground-truth values, matching the input type of `estimates`.
        If a dict, per-key arrays. If an array, shape
        (num_sims, num_params) directly.
    variable_keys  : sequence of str or None, optional, default: None
        Which keys to select and plot, and in what order, when
        `estimates`/`targets` are dicts. By default, all keys, in
        dict insertion order. Ignored for array input.
    variable_names : sequence of str or None, optional, default: None
        Display names for the plotted columns. Defaults to
        `variable_keys` (dict input) or `param_0`, `param_1`, ...
        (array input).
    color          : str, optional, default: "#822621"
        Base color for the calibration ECDF lines.
    title_fontsize : int, optional, default: 22
        The font size of the panel titles.
    label_fontsize : int, optional, default: 18
        The font size of the axis label texts.
    metric_fontsize : int, optional, default: 18
        The font size of the displayed calibration metric text.
    tick_fontsize  : int, optional, default: 16
        The font size of the axis tick labels.
    **kwargs
        Forwarded to `bf.diagnostics.plots.calibration_ecdf` (e.g.
        `figsize`, `num_row`, `num_col`).

    Returns
    -------
    fig : plt.Figure - the calibration diagnostic figure
    """
    estimates_arr, targets_arr, names = prepare_plot_data(estimates, targets, variable_keys, variable_names)

    return bf.diagnostics.plots.calibration_ecdf(
        estimates=estimates_arr,
        targets=targets_arr,
        variable_names=names,
        rank_ecdf_color=color,
        title_fontsize=title_fontsize,
        label_fontsize=label_fontsize,
        metric_fontsize=metric_fontsize,
        tick_fontsize=tick_fontsize,
        **kwargs,
    )
