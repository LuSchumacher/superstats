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
    BASE_COL_WIDTH,
    BASE_ROW_HEIGHT,
    HSPACE,
    LABEL_FONTSIZE,
    LABEL_PAD,
    TICK_FONTSIZE,
    TITLE_FONTSIZE,
    WSPACE,
    Y_LABEL_PAD,
)
from superstats.utils.indexing import format_dataset_label, normalize_data_indices
from superstats.utils.plotting import (
    compute_uncertainty_bands,
    get_default_num_cols,
    get_layout,
    get_uncertainty_band_label,
    plot_dist,
    plot_uncertainty_bands,
    resolve_dist_alpha,
    smooth_trajectories,
)


def plot_time_varying_posterior(
    estimates: Mapping[str, np.ndarray] | np.ndarray,
    targets: Mapping[str, np.ndarray] | np.ndarray | None = None,
    variable_keys: Sequence[str] | None = None,
    variable_names: Sequence[str] | None = None,
    aggregation: Callable | None = None,
    aggregate_strategy: Literal["full_uncertainty", "no_epistemic"] = "full_uncertainty",
    uncertainty_fun: Literal["std", "ci", "mad", "hdi"] | Callable | None = "ci",
    smoothing: Literal["sma", "ema"] | None = None,
    smoothing_window: int = 5,
    marginal: bool = True,
    dist_type: Literal["hist", "kde", "both"] = "hist",
    num_bins: int | None = None,
    dist_alpha: float | None = None,
    num_cols: int | None = None,
    alpha: float = 0.5,
    color: str = BASE_COLOR,
    title_fontsize: int = TITLE_FONTSIZE,
    label_fontsize: int = LABEL_FONTSIZE,
    tick_fontsize: int = TICK_FONTSIZE,
    figsize: tuple[float, float] | None = None,
    data_idx: int | Sequence[int] | None = None,
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
        "full_uncertainty": flatten only the dataset and posterior-sample
        axes into one trajectory pool. The ribbon contains posterior and
        between-dataset variation.
        "no_epistemic": take the posterior median within each dataset,
        preserving the dataset axis. The ribbon then contains only
        between-dataset variation.
    uncertainty_fun    : {"std", "ci", "mad", "hdi"} or callable or None, optional, default: "ci"
        Named methods draw nested outer/inner ribbons: ±1/±0.5 SD,
        95%/65% CI, ±1.48/±0.74 MAD, or 95%/65% HDI. A callable
        receives (N, T) trajectories and draws the single `(lo, hi)`
        interval it returns, with each bound shaped (T,).
    smoothing          : {"sma", "ema"} or None, optional, default: None
        Applied to each trajectory (and to `targets`, if given) before
        computing the center, uncertainty, and marginal.
    smoothing_window   : int, optional, default: 5
        Window size for `sma`, or span parameter for `ema`.
    marginal           : bool, optional, default: True
        Attach a marginal distribution panel to the right of each
        time-series axis. The distribution is computed from the exact same
        strategy-specific trajectory pool used for the uncertainty band.
    dist_type          : {"hist", "kde", "both"}, optional, default: "hist"
        Distribution type used for marginal panels.
    num_bins           : int or None, optional, default: None
        Number of histogram bins. If None, Seaborn selects the bins.
    dist_alpha         : float or None, optional, default: None
        Opacity of marginal distributions. If None, uses 1.0 for a
        single distribution and 0.5 when targets are overlaid.
    num_cols           : int or None, optional, default: None
        Exact number of grid columns. If None, non-aggregated plots use
        one column per selected dataset and aggregated plots use the
        shared compact dynamic layout.
    color              : str, optional, default: BASE_COLOR
        Line and band color.
    alpha              : float in [0, 1], optional, default: 0.5
        Alpha for the darker inner uncertainty ribbon. The outer ribbon
        uses half this opacity.
    title_fontsize     : int, optional, default: 22
        The font size of the panel titles.
    label_fontsize     : int, optional, default: 18
        The font size of the axis label texts.
    tick_fontsize      : int, optional, default: 16
        The font size of the axis tick labels.
    figsize            : tuple of two floats or None, optional, default: None
        Explicit figure size in inches. If None, the default layout size
        is used.
    data_idx           : int, sequence of int, or None, optional, default: None
        Dataset indices to plot. None selects all datasets. A single
        integer preserves the dataset axis, and a sequence preserves
        the requested order.

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

    num_datasets = next(iter(local_estimates.values())).shape[0]
    selected_indices = normalize_data_indices(data_idx, num_datasets)
    local_estimates = {name: values[selected_indices] for name, values in local_estimates.items()}
    if local_targets is not None:
        local_targets = {name: values[selected_indices] for name, values in local_targets.items()}

    D, S, T = next(iter(local_estimates.values())).shape
    P = len(names)
    t = np.arange(T)

    # layout
    if aggregation is None:
        matrix_layout = num_cols is None
        if matrix_layout:
            num_rows = P
            layout_num_cols = D
        else:
            layout_num_cols = num_cols
            num_rows = int(np.ceil(P * D / layout_num_cols))
        col_width = BASE_COL_WIDTH
        row_height = BASE_ROW_HEIGHT
    else:
        matrix_layout = False
        if num_cols is None:
            num_cols = get_default_num_cols(P)
        num_rows = int(np.ceil(P / num_cols))
        layout_num_cols = num_cols
        col_width = BASE_COL_WIDTH
        row_height = BASE_ROW_HEIGHT

    plot_figsize, legend_bottom, legend_y = get_layout(
        num_rows,
        layout_num_cols,
        figsize,
        col_width=col_width,
        row_height=row_height,
    )
    fig, axes = plt.subplots(
        num_rows,
        layout_num_cols,
        figsize=plot_figsize,
        squeeze=False,
    )

    axes_flat = axes.ravel()

    # per-panel loop
    panel = 0
    has_uncertainty_band = False
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

            trajectories = smooth_trajectories(trajectories, smoothing, smoothing_window)

            # center
            if aggregation is None:
                center = np.median(trajectories, axis=0)
            else:
                center = np.asarray(aggregation(trajectories, axis=0))

            # uncertainty bands
            uncertainty_bands = None
            if uncertainty_fun is not None:
                uncertainty_bands = compute_uncertainty_bands(
                    trajectories,
                    uncertainty_fun,
                    center,
                )

            # target trajectory (optional)
            target_line = None
            if local_targets is not None:
                if aggregation is None:
                    target_line = smooth_trajectories(
                        local_targets[name][d : d + 1, :],
                        smoothing,
                        smoothing_window,
                    )[0]
                else:
                    smoothed_targets = smooth_trajectories(
                        local_targets[name],
                        smoothing,
                        smoothing_window,
                    )
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
            if uncertainty_bands is not None:
                has_uncertainty_band |= plot_uncertainty_bands(
                    ax,
                    t,
                    uncertainty_bands[0],
                    uncertainty_bands[1],
                    color,
                    alpha=alpha,
                )
            ax.plot(t, center, color=color, linewidth=1.5)
            if target_line is not None:
                ax.plot(t, target_line, color="black", linewidth=1.5, linestyle="--", zorder=5)

            if aggregation is None:
                if matrix_layout:
                    if panel < D:
                        ax.set_title(
                            format_dataset_label(selected_indices[d]),
                            fontsize=title_fontsize,
                            pad=15,
                        )
                    if d == 0:
                        ax.set_ylabel(
                            name,
                            fontsize=label_fontsize,
                            rotation=0,
                            labelpad=Y_LABEL_PAD,
                        )
                    if panel >= (P - 1) * D:
                        ax.set_xlabel(
                            "Step",
                            fontsize=label_fontsize,
                            labelpad=LABEL_PAD,
                        )
                else:
                    title = name if D == 1 else f"{name} — {format_dataset_label(selected_indices[d])}"
                    ax.set_title(
                        title,
                        fontsize=title_fontsize,
                        pad=15,
                    )
                    ax.set_ylabel(
                        "Parameter value" if panel % layout_num_cols == 0 else "",
                        fontsize=label_fontsize,
                        labelpad=Y_LABEL_PAD,
                    )
                    if panel // layout_num_cols == num_rows - 1:
                        ax.set_xlabel(
                            "Step",
                            fontsize=label_fontsize,
                            labelpad=LABEL_PAD,
                        )
            else:
                ax.set_title(name, fontsize=title_fontsize, pad=15)
                if panel // layout_num_cols == num_rows - 1:
                    ax.set_xlabel(
                        "Step",
                        fontsize=label_fontsize,
                        labelpad=LABEL_PAD,
                    )
                ax.set_ylabel(
                    "Parameter value" if panel % layout_num_cols == 0 else "",
                    fontsize=label_fontsize,
                    labelpad=Y_LABEL_PAD,
                )

            ax.tick_params(labelsize=tick_fontsize)
            ax.grid(alpha=0.3)

            # marginal distribution
            if marginal:
                panel_dist_alpha = resolve_dist_alpha(
                    dist_alpha,
                    2 if target_line is not None else 1,
                )
                plot_dist(
                    trajectories.reshape(-1),
                    ax=ax_kde,
                    dist_type=dist_type,
                    color=color,
                    orientation="vertical",
                    num_bins=num_bins,
                    alpha=panel_dist_alpha,
                    hide_axis=True,
                )
                if target_line is not None:
                    target_marginal_ax = ax_kde.twiny() if dist_type == "hist" else ax_kde
                    plot_dist(
                        target_line.reshape(-1),
                        ax=target_marginal_ax,
                        dist_type=dist_type,
                        color="black",
                        orientation="vertical",
                        num_bins=num_bins,
                        alpha=panel_dist_alpha,
                        hide_axis=True,
                    )
                    target_marginal_ax.set_ylim(ax.get_ylim())
                    target_marginal_ax.set_axis_off()
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
    if has_uncertainty_band:
        band_label = get_uncertainty_band_label(uncertainty_fun)
        handles.append(mpatches.Patch(color=color, alpha=alpha, label=band_label))
    if local_targets is not None:
        handles.append(mlines.Line2D([], [], color="black", linewidth=1.5, linestyle="--", label="Target"))

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        fontsize=label_fontsize,
        framealpha=0.0,
        bbox_to_anchor=(0.5, legend_y),
    )
    sns.despine()
    plt.tight_layout()
    fig.subplots_adjust(
        bottom=legend_bottom,
        hspace=HSPACE,
        wspace=WSPACE,
    )
    return fig
