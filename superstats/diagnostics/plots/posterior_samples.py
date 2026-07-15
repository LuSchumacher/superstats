"""Posterior sample visualization helpers."""

from collections.abc import Callable, Mapping, Sequence
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Literal

from superstats.defaults import (
    BASE_COLOR,
    CATEGORICAL_PALETTE,
)

plt.rcParams["axes.axisbelow"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Palatino", "Palatino Linotype", "DejaVu Serif"]

BASE_COL_WIDTH = 6.0
BASE_ROW_HEIGHT = 3.0

BAND_LABELS = {
    "std": "±1 SD",
    "95ci": "95% CI",
    "mad": "±1.48 MAD",
    "95hdi": "95% HDI",
}


def _smooth_trajectories(
    arr: np.ndarray,
    smoothing: Literal["sma", "ema"] | None,
    smoothing_window: int = 5,
) -> np.ndarray:
    """Apply SMA/EMA smoothing along the last axis of an array.

    Parameters
    ----------
    arr              : np.ndarray of shape (..., T)
        Trajectories to smooth, smoothed independently along the last
        axis for each leading index.
    smoothing        : {"sma", "ema"} or None
        Smoothing method. If None, `arr` is returned unchanged.
    smoothing_window : int, optional, default: 5
        Window size for `sma`, or span parameter for `ema`.

    Returns
    -------
    smoothed : np.ndarray of the same shape as `arr`
    """
    if smoothing is None:
        return arr
    T = arr.shape[-1]
    smoothed = arr.copy()
    if smoothing == "sma":
        for i in range(T):
            smoothed[..., i] = arr[..., max(0, i - smoothing_window + 1) : i + 1].mean(axis=-1)
    elif smoothing == "ema":
        a = 2.0 / (smoothing_window + 1)
        for i in range(1, T):
            smoothed[..., i] = a * arr[..., i] + (1 - a) * smoothed[..., i - 1]
    return smoothed


def plot_time_varying_posterior(
    estimates: Mapping[str, np.ndarray] | np.ndarray,
    targets: Mapping[str, np.ndarray] | np.ndarray | None = None,
    variable_keys: Sequence[str] | None = None,
    variable_names: Sequence[str] | None = None,
    aggregation: Callable | None = None,
    aggregate_strategy: Literal["full_uncertainty", "no_epistemic"] = "full_uncertainty",
    uncertainty_fun: Literal["std", "95ci", "mad", "95hdi"] | Callable | None = "95ci",
    smoothing: Literal["sma", "ema"] | None = None,
    smoothing_window: int = 5,
    marginal: bool = True,
    num_cols: int = 2,
    alpha: float = 0.5,
    color: str = BASE_COLOR,
    title_fontsize: int = 22,
    label_fontsize: int = 18,
    tick_fontsize: int = 16,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Plot time-varying parameter posteriors.

    Parameters
    ----------
    estimates          : Mapping[str, np.ndarray] or np.ndarray
        Posterior samples. If a dict, values of shape
        (num_datasets, num_post_samples, num_steps, 1), keyed by
        variable. If an array, shape
        (num_datasets, num_post_samples, num_steps, num_params)
        directly.
    targets            : Mapping[str, np.ndarray], np.ndarray, or None, optional, default: None
        Ground-truth trajectories, matching the input type of
        `estimates`. If a dict, values of shape
        (num_datasets, num_steps, 1). If an array, shape
        (num_datasets, num_steps, num_params) directly. If given,
        drawn as a black dashed line on top of each panel: the raw
        per-dataset trajectory when `aggregation` is None, or
        aggregated across datasets (using `aggregation`) when
        `aggregation` is not None. Smoothed with the same `smoothing`
        settings as the posterior trajectories, for a fair visual
        comparison.
    variable_keys      : sequence of str or None, optional, default: None
        Which variables to select and plot, and in what order, when
        `estimates`/`targets` are dicts. By default, all keys, in
        dict insertion order. Ignored for array input.
    variable_names     : sequence of str or None, optional, default: None
        Display names (used for panel labels/titles), in the same
        order as `variable_keys` (or the array's last axis). Defaults
        to `variable_keys` for dict input, or `param_0`, `param_1`,
        ... for array input.
    aggregation        : callable or None, optional, default: None
        None: one panel per (param, dataset).
        callable: one panel per param, aggregated across datasets.
        Called as `aggregation(trajectories, axis=0)` and must return
        a (T,) center. The same function aggregates `targets` across
        datasets when both `targets` and `aggregation` are given.
    aggregate_strategy : {"full_uncertainty", "no_epistemic"}, optional, default: "full_uncertainty"
        Only used when `aggregation` is not None.
        "full_uncertainty": flatten datasets and posterior samples,
        then summarize.
        "no_epistemic": median across posterior samples per dataset
        first, then aggregate.
    uncertainty_fun    : {"std", "95ci", "mad", "95hdi"} or callable or None, optional, default: "95ci"
        Band drawn around the center line. A callable receives (N, T)
        trajectories and must return `(lo, hi)`, each of shape (T,).
    smoothing          : {"sma", "ema"} or None, optional, default: None
        Applied to each trajectory (and to `targets`, if given) before
        computing the center, uncertainty, and marginal.
    smoothing_window   : int, optional, default: 5
        Window size for `sma`, or span parameter for `ema`.
    marginal           : bool, optional, default: True
        Attach a marginal KDE panel to the right of each trajectory
        axis. The KDE is computed on the same array used for the
        uncertainty band.
    num_cols           : int, optional, default: 2
        Number of subplot columns when `aggregation` is not None.
    color              : str, optional, default: BASE_COLOR
        Line and band color.
    alpha              : float in [0, 1], optional, default: 0.5
        Alpha for the uncertainty band.
    title_fontsize     : int, optional, default: 22
        The font size of the panel titles.
    label_fontsize     : int, optional, default: 18
        The font size of the axis label texts.
    tick_fontsize      : int, optional, default: 16
        The font size of the axis tick labels.
    figsize            : tuple of two floats or None, optional, default: None
        Explicit figure size in inches. If None, the default layout size
        is used.

    Returns
    -------
    fig : plt.Figure - the figure instance for optional saving

    Raises
    ------
    ValueError
        If `estimates`/`targets` are inconsistent (mismatched dict
        keys, or a `variable_names` length mismatch for array input),
        or if `aggregate_strategy`, or `uncertainty_fun` when given as
        a string, is not one of the recognized values.
    """
    if isinstance(estimates, Mapping):
        keys = list(variable_keys) if variable_keys is not None else list(estimates.keys())
        missing = [k for k in keys if k not in estimates]
        if missing:
            raise ValueError(f"variable_keys not found in estimates: {missing}")
        if targets is not None:
            missing_t = [k for k in keys if k not in targets]
            if missing_t:
                raise ValueError(f"variable_keys not found in targets: {missing_t}")

        names = list(variable_names) if variable_names is not None else keys
        local_estimates = {n: estimates[k][..., 0] for k, n in zip(keys, names)}
        local_targets = {n: targets[k][..., 0] for k, n in zip(keys, names)} if targets is not None else None
    else:
        num_params = estimates.shape[-1]
        names = list(variable_names) if variable_names is not None else [f"param_{p}" for p in range(num_params)]
        if len(names) != num_params:
            raise ValueError(f"variable_names has {len(names)} entries but there are {num_params} variables.")

        local_estimates = {n: estimates[..., p] for p, n in enumerate(names)}
        local_targets = {n: targets[..., p] for p, n in enumerate(names)} if targets is not None else None

    D, S, T = next(iter(local_estimates.values())).shape
    P = len(names)
    t = np.arange(T)

    # layout
    if aggregation is None:
        default_figsize = (BASE_COL_WIDTH * D, BASE_ROW_HEIGHT * P)
        fig, axes = plt.subplots(P, D, figsize=figsize if figsize is not None else default_figsize, squeeze=False)
    else:
        num_rows = int(np.ceil(P / num_cols))
        col_width = BASE_COL_WIDTH * (1.3 if marginal else 1.0)
        row_height = BASE_ROW_HEIGHT * 1.6
        default_figsize = (col_width * num_cols, row_height * num_rows)
        fig, axes = plt.subplots(
            num_rows,
            num_cols,
            figsize=figsize if figsize is not None else default_figsize,
            squeeze=False,
        )

    axes_flat = axes.ravel()

    # per-panel loop
    panel = 0
    for name in names:
        datasets = range(D) if aggregation is None else [None]
        for d in datasets:
            # prepare trajectories (N, T)
            if aggregation is None:
                trajectories = local_estimates[name][d, :, :]
            else:
                param = local_estimates[name]
                if aggregate_strategy == "full_uncertainty":
                    trajectories = param.reshape(D * S, T)
                elif aggregate_strategy == "no_epistemic":
                    trajectories = np.median(param, axis=1)
                else:
                    raise ValueError(f"Unknown aggregate_strategy: {aggregate_strategy!r}")

            trajectories = _smooth_trajectories(trajectories, smoothing, smoothing_window)

            # center
            if aggregation is None:
                center = np.median(trajectories, axis=0)
            else:
                center = np.asarray(aggregation(trajectories, axis=0))

            # uncertainty bands
            lo, hi = None, None
            if callable(uncertainty_fun):
                lo, hi = uncertainty_fun(trajectories)
                lo, hi = np.asarray(lo), np.asarray(hi)
            elif uncertainty_fun == "std":
                sd = trajectories.std(axis=0)
                lo, hi = center - sd, center + sd
            elif uncertainty_fun == "95ci":
                lo, hi = np.percentile(trajectories, 2.5, axis=0), np.percentile(trajectories, 97.5, axis=0)
            elif uncertainty_fun == "mad":
                mad = np.median(np.abs(trajectories - center), axis=0)
                lo, hi = center - 1.4826 * mad, center + 1.4826 * mad
            elif uncertainty_fun == "95hdi":
                lo, hi = np.empty(T), np.empty(T)
                for i in range(T):
                    vals = np.sort(trajectories[:, i])
                    n = len(vals)
                    window = int(np.floor(0.95 * n))
                    widths = vals[window:] - vals[: n - window]
                    idx = np.argmin(widths)
                    lo[i], hi[i] = vals[idx], vals[idx + window]
            elif uncertainty_fun is not None:
                raise ValueError(f"Unknown uncertainty_fun: {uncertainty_fun!r}")

            # target trajectory (optional)
            target_line = None
            if local_targets is not None:
                if aggregation is None:
                    target_line = _smooth_trajectories(local_targets[name][d : d + 1, :], smoothing, smoothing_window)[
                        0
                    ]
                else:
                    smoothed_targets = _smooth_trajectories(local_targets[name], smoothing, smoothing_window)
                    target_line = np.asarray(aggregation(smoothed_targets, axis=0))

            # axes setup
            ax_base = axes_flat[panel]
            if marginal:
                sub = ax_base.get_subplotspec().subgridspec(1, 2, width_ratios=[4.2, 0.8], wspace=0.0)
                ax = fig.add_subplot(sub[0])
                ax_kde = fig.add_subplot(sub[1])
                ax_base.axis("off")
            else:
                ax = ax_base
                ax_kde = None

            # plot
            if lo is not None:
                ax.fill_between(t, lo, hi, color=color, alpha=alpha, edgecolor="none")
            ax.plot(t, center, color=color, linewidth=1.5)
            if target_line is not None:
                ax.plot(t, target_line, color="black", linewidth=1.5, linestyle="--", zorder=5)

            if aggregation is None:
                if panel < D:
                    ax.set_title(f"Dataset {d}", fontsize=title_fontsize, pad=15)
                if d == 0:
                    ax.set_ylabel(name, fontsize=label_fontsize, rotation=0, labelpad=20)
                if panel >= (P - 1) * D:
                    ax.set_xlabel("Step", fontsize=label_fontsize)
            else:
                ax.set_title(name, fontsize=title_fontsize, pad=15)
                if panel // num_cols == num_rows - 1:
                    ax.set_xlabel("Step", fontsize=label_fontsize)

            ax.tick_params(labelsize=tick_fontsize)
            ax.grid(alpha=0.3)

            # marginal KDE
            if marginal:
                sns.kdeplot(y=trajectories.ravel(), ax=ax_kde, color=color, fill=True, alpha=1)
                ax_kde.set_ylim(ax.get_ylim())
                ax_kde.set_axis_off()

            panel += 1

    for j in range(panel, len(axes_flat)):
        axes_flat[j].axis("off")

    # legend
    if aggregation is not None:
        aggregate_label = getattr(aggregation, "__name__", "aggregate").capitalize()
    else:
        aggregate_label = "Median"
    handles = [mlines.Line2D([], [], color=color, linewidth=1.5, label=aggregate_label)]
    if uncertainty_fun is not None:
        band_label = "Uncertainty band" if callable(uncertainty_fun) else BAND_LABELS[uncertainty_fun]
        handles.append(mpatches.Patch(color=color, alpha=alpha, label=band_label))
    if local_targets is not None:
        handles.append(mlines.Line2D([], [], color="black", linewidth=1.5, linestyle="--", label="Target"))

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        fontsize=label_fontsize,
        framealpha=0.0,
        bbox_to_anchor=(0.5, -0.02),
    )
    sns.despine()
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    return fig


def plot_time_invariant_posterior(
    estimates: Mapping[str, np.ndarray] | np.ndarray,
    targets: Mapping[str, np.ndarray] | np.ndarray | None = None,
    variable_keys: Sequence[str] | None = None,
    variable_names: Sequence[str] | None = None,
    aggregation: Callable | None = None,
    mixture_names: dict | None = None,
    num_cols: int = 2,
    color: str = BASE_COLOR,
    title_fontsize: int = 22,
    label_fontsize: int = 18,
    tick_fontsize: int = 16,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Plot time-invariant parameter posteriors.

    Parameters
    ----------
    estimates      : Mapping[str, np.ndarray] or np.ndarray
        Posterior samples. If a dict, values of shape
        (num_datasets, num_post_samples, num_steps, num_components),
        keyed by variable. If an array, shape
        (num_datasets, num_post_samples, num_steps, num_params)
        directly - treated as single-component parameters; mixture
        grouping is not inferable from array input.
    targets        : Mapping[str, np.ndarray], np.ndarray, or None, optional, default: None
        Ground-truth values, matching the input type of `estimates`.
        If a dict, values of shape (num_datasets, num_components). If
        an array, shape (num_datasets, num_params) directly. If
        given, drawn as black dashed vertical lines. When
        `aggregation` is None, one solid line per panel marks that
        panel's specific dataset's true value. When `aggregation` is
        given, the per-dataset true values are collapsed with
        `aggregation` and a single solid line is drawn per panel.
    variable_keys  : sequence of str or None, optional, default: None
        Which variables to select and plot, and in what order, when
        `estimates` is a dict. By default, all keys, in dict
        insertion order. Ignored for array input.
    variable_names : sequence of str or None, optional, default: None
        Display names (used for panel labels/titles), in the same
        order as `variable_keys` (or the array's last axis). Defaults
        to `variable_keys` for dict input, or `param_0`, `param_1`,
        ... for array input.
    aggregation    : callable or None, optional, default: None
        Controls both the posterior layout and the target summary.
        If None: one panel per (dataset, parameter) pair; rows=params,
        cols=datasets, param name as row label, dataset index as
        column title; `targets` (if given) are shown per dataset.
        If a callable (e.g. np.mean, np.median): posterior samples are
        pooled across datasets into one panel per parameter, arranged
        in a `num_cols`-column grid; `targets` (if given) are
        collapsed across datasets with `aggregation` into a single
        reference value per panel.
    mixture_names  : dict or None, optional, default: None
        Mapping from base parameter name (e.g. "a", without any
        "_mixture_weights" suffix) to a list of component names.
        Defaults to "component 0", "component 1", ... when not
        supplied. Only applies to dict input with multi-component
        values.
    num_cols       : int, optional, default: 2
        Number of subplot columns when `aggregation` is not None.
    color          : str, optional, default: BASE_COLOR
        Base color for non-mixture parameters.
    title_fontsize : int, optional, default: 22
        The font size of the panel titles.
    label_fontsize : int, optional, default: 18
        The font size of the row labels (param names, non-pooled
        layout only).
    tick_fontsize  : int, optional, default: 16
        The font size of the axis tick labels.
    figsize        : tuple of two floats or None, optional, default: None
        Explicit figure size in inches. If None, the default layout size
        is used.

    Returns
    -------
    fig : plt.Figure - the figure instance for optional saving

    Raises
    ------
    ValueError
        If no variables are found to plot (empty `variable_keys`,
        whether resolved by default or passed explicitly), or if
        `variable_names` doesn't match the number of variables for
        array input.
    """
    mixture_names = mixture_names or {}

    if isinstance(estimates, Mapping):
        keys = list(variable_keys) if variable_keys is not None else list(estimates.keys())
        if not keys:
            raise ValueError("No variables found to plot.")
        missing = [k for k in keys if k not in estimates]
        if missing:
            raise ValueError(f"variable_keys not found in estimates: {missing}")

        names = list(variable_names) if variable_names is not None else keys
        local_estimates = {n: estimates[k] for k, n in zip(keys, names)}
        local_mixture_names = {
            n: mixture_names[k.split("_mixture_weights")[0]]
            for k, n in zip(keys, names)
            if k.split("_mixture_weights")[0] in mixture_names
        }
        local_targets = {n: targets[k] for k, n in zip(keys, names)} if targets is not None else None
    else:
        num_params = estimates.shape[-1]
        names = list(variable_names) if variable_names is not None else [f"param_{p}" for p in range(num_params)]
        if len(names) != num_params:
            raise ValueError(f"variable_names has {len(names)} entries but there are {num_params} variables.")
        if not names:
            raise ValueError("No variables found to plot.")

        local_estimates = {n: estimates[..., p : p + 1] for p, n in enumerate(names)}
        local_mixture_names = {}
        local_targets = {n: targets[..., p : p + 1] for p, n in enumerate(names)} if targets is not None else None

    D = next(iter(local_estimates.values())).shape[0]

    # panels meta
    panels_meta = []
    for name in names:
        n_components = local_estimates[name].shape[-1]
        if n_components > 1:
            comp_names = local_mixture_names.get(name, [f"component {i}" for i in range(n_components)])
            panels_meta.append((name, list(range(n_components)), comp_names, True))
        else:
            panels_meta.append((name, [0], [name], False))

    P = len(panels_meta)

    # pool across samples and steps
    pooled = {}
    for name in names:
        arr = local_estimates[name]
        B, S, T, C = arr.shape
        for c in range(C):
            pooled[(name, c)] = arr[:, :, :, c].reshape(B, S * T)

    # layout
    if aggregation is None:
        default_figsize = (BASE_COL_WIDTH * D, BASE_ROW_HEIGHT * P)
        fig, axes = plt.subplots(P, D, figsize=figsize if figsize is not None else default_figsize, squeeze=False)
    else:
        num_rows = int(np.ceil(P / num_cols))
        default_figsize = (BASE_COL_WIDTH * num_cols, BASE_ROW_HEIGHT * num_rows)
        fig, axes = plt.subplots(
            num_rows,
            num_cols,
            figsize=figsize if figsize is not None else default_figsize,
            squeeze=False,
        )

    axes_flat = axes.ravel()

    # per-panel loop
    panel = 0
    legend_drawn = False
    target_legend_drawn = False
    for p, (param_name, comp_indices, comp_labels, is_mixture) in enumerate(panels_meta):
        datasets = range(D) if aggregation is None else [None]
        for d in datasets:
            ax = axes_flat[panel]

            for ci, (c, label) in enumerate(zip(comp_indices, comp_labels)):
                c_color = CATEGORICAL_PALETTE[ci % len(CATEGORICAL_PALETTE)] if is_mixture else color
                if aggregation is None:
                    sns.kdeplot(
                        x=pooled[(param_name, c)][d],
                        ax=ax,
                        color=c_color,
                        fill=True,
                        alpha=1,
                        label=label if is_mixture else None,
                    )
                else:
                    sns.kdeplot(
                        x=pooled[(param_name, c)].ravel(),
                        ax=ax,
                        color=c_color,
                        fill=True,
                        alpha=1,
                        label=label if is_mixture else None,
                    )

                if local_targets is not None:
                    target_vals = local_targets[param_name][:, c]

                    if aggregation is None:
                        target_val = target_vals[d]
                    else:
                        target_val = aggregation(target_vals, axis=0)

                    ax.axvline(
                        target_val,
                        color="black",
                        linestyle="--",
                        linewidth=1.5,
                        zorder=5,
                        label="Target" if not target_legend_drawn else None,
                    )
                    target_legend_drawn = True

            if aggregation is None:
                if p == 0:
                    ax.set_title(f"Dataset {d}", fontsize=title_fontsize, pad=15)
                if d == 0:
                    ax.set_ylabel(param_name, fontsize=label_fontsize, rotation=0, labelpad=80)
                else:
                    ax.set_ylabel("")
            else:
                ax.set_title(param_name, fontsize=title_fontsize, pad=15)
                ax.set_ylabel("")

            ax.set_xlabel("")

            if (is_mixture or local_targets is not None) and not legend_drawn:
                ax.legend(fontsize=tick_fontsize, framealpha=0.0)
                legend_drawn = True

            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=tick_fontsize)
            panel += 1

    for j in range(panel, len(axes_flat)):
        axes_flat[j].axis("off")

    sns.despine()
    plt.tight_layout()
    return fig
