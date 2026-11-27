"""Parameter plots with context held fixed for every complete trajectory."""

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from superstats.defaults import (
    BASE_COLOR,
    CATEGORICAL_PALETTE,
    LABEL_FONTSIZE,
    TICK_FONTSIZE,
    TITLE_FONTSIZE,
    UNCERTAINTY_LABELS,
)
from superstats.utils.plotting import (
    compute_uncertainty_bands,
    format_condition_label,
    get_default_num_cols,
    plot_dist,
    plot_uncertainty_bands,
    smooth_trajectories,
)


def plot_conditional_parameters(
    draws,
    prior=False,
    num_trajectories=0,
    marginal=True,
    num_cols=None,
    dist_type="hist",
    num_bins=None,
    dist_alpha=None,
    alpha=0.35,
    color=BASE_COLOR,
    colors=None,
    targets=None,
    aggregation=None,
    uncertainty_fun="hdi",
    smoothing=None,
    smoothing_window=5,
    variable_names=None,
    figsize=None,
    title_fontsize=TITLE_FONTSIZE,
    label_fontsize=LABEL_FONTSIZE,
    tick_fontsize=TICK_FONTSIZE,
    parameter_groups=None,
):
    """Plot parameter trajectories under fixed predictor conditions.

    Parameters
    ----------
    draws              : ConditionalDraws
        Prepared parameter samples. Each entry in `parameters` has shape
        (num_datasets, num_draws, num_conditions, num_steps). `data_indices`
        identifies the selected datasets. Conditions come from `conditions`
        or, when provided, the per-parameter `parameter_conditions` mapping.
    prior              : bool, optional, default: False
        If True, show all individual prior draws and their arithmetic mean.
        Otherwise, show posterior medians and uncertainty intervals.
    num_trajectories   : int, optional, default: 0
        Number of individual posterior draws to overlay, capped at the
        available draw count. Must be non-negative. Ignored in prior mode,
        which displays all supplied draws.
    marginal           : bool, optional, default: True
        Attach a marginal distribution to the right of each trajectory
        panel, using the same smoothed draws as the trajectory panel.
    num_cols           : int or None, optional, default: None
        Exact number of grid columns. If None, use a compact layout based
        on the number of parameters in prior mode, or one column per
        dataset in posterior mode. With `parameter_groups`, use enough
        columns for the largest group. Must be positive.
    dist_type          : {"hist", "kde", "both"}, optional, default: "hist"
        Distribution type used for marginal panels.
    num_bins           : int or None, optional, default: None
        Number of histogram bins. If None, Seaborn selects the bins.
    dist_alpha         : float or None, optional, default: None
        Opacity of sample marginal distributions. If None, use 1.0 for a
        single condition and 0.45 for multiple conditions.
    alpha              : float in [0, 1], optional, default: 0.35
        Opacity of individual trajectories and inner uncertainty ribbons.
        Outer ribbons use half this opacity.
    color              : matplotlib color, optional, default: BASE_COLOR
        Trajectory and band color when only one condition is plotted and
        `colors` is None.
    colors             : sequence of matplotlib colors or None, optional, default: None
        One color per condition. Must cover the largest condition count
        across parameters. If None, use `color` for one condition or a
        categorical palette for multiple conditions.
    targets            : ConditionalDraws or None, optional, default: None
        Prepared ground-truth trajectories with shape
        (num_datasets, 1, num_conditions, num_steps) per parameter, matching
        the parameters, dataset order, and conditions in `draws`. Drawn as
        dashed lines and included in marginal panels when enabled.
    aggregation        : callable or None, optional, default: None
        If None, show one panel per parameter and dataset. Otherwise,
        aggregate draws across datasets with `aggregation(values, axis=0)`,
        then use the same function across draws to compute the center.
        Targets are aggregated across datasets with the same function.
    uncertainty_fun    : {"std", "ci", "mad", "hdi"} or callable or None, optional, default: "hdi"
        Posterior uncertainty method. Named methods draw nested outer/inner
        ribbons: +/-1/+/-0.5 SD, 95%/65% CI, +/-1.48/+/-0.74 MAD, or
        95%/65% HDI. A callable receives (num_draws, num_steps) trajectories
        and returns a single (lower, upper) interval, each bound shaped
        (num_steps,). None disables ribbons. Ignored in prior mode.
    smoothing          : {"sma", "ema"} or None, optional, default: None
        Smooth each trajectory and target before computing centers,
        uncertainty intervals, and marginal distributions.
    smoothing_window   : int, optional, default: 5
        Window size for `sma`, or span parameter for `ema`.
    variable_names     : sequence of str or None, optional, default: None
        Display names in the insertion order of `draws.parameters`.
        Must contain one name per plotted parameter. If None, use the keys.
    figsize            : tuple of two floats or None, optional, default: None
        Explicit figure size in inches. If None, size the figure according
        to the panel grid and figure legend.
    title_fontsize     : int, optional, default: 22
        Font size of panel titles.
    label_fontsize     : int, optional, default: 18
        Font size of axis labels.
    tick_fontsize      : int, optional, default: 16
        Font size of tick labels. The figure legend uses this size capped
        at 11.
    parameter_groups   : sequence of sequences of str or None, optional, default: None
        Panel keys grouped into parameter rows, with one row per dataset.
        Groups exceeding `num_cols` continue on additional rows.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.

    Raises
    ------
    ValueError
        If `num_trajectories`, `dist_type`, `alpha`, `colors`, `num_cols`,
        or the number of display names is invalid.

    Notes
    -----
    Links and formulas must be applied before calling this renderer. Draws
    remain paired across conditions; this function does not resample them.
    Condition legends are hidden. The figure legend describes centers,
    individual draws, uncertainty intervals, and targets when present.
    """
    if (
        isinstance(num_trajectories, bool)
        or not isinstance(num_trajectories, (int, np.integer))
        or num_trajectories < 0
    ):
        raise ValueError("num_trajectories must be a non-negative integer.")
    if dist_type not in {"hist", "kde", "both"}:
        raise ValueError("dist_type must be 'hist', 'kde', or 'both'.")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between zero and one.")
    condition_count = max(values.shape[2] for values in draws.parameters.values())
    if colors is None:
        colors = [color] if condition_count == 1 else CATEGORICAL_PALETTE
        if len(colors) < condition_count:
            colors = [plt.get_cmap("tab20")(i % 20) for i in range(condition_count)]
    if isinstance(colors, str) or len(colors) < condition_count:
        raise ValueError("colors must supply at least one color per condition.")
    names = list(draws.parameters)
    display_names = names if variable_names is None else list(variable_names)
    if len(display_names) != len(names):
        raise ValueError("variable_names must have one name per plotted parameter.")
    num_datasets = len(draws.data_indices)
    if aggregation is not None:
        num_datasets = 1
    default_columns = (
        max(map(len, parameter_groups))
        if parameter_groups is not None
        else (get_default_num_cols(len(names)) if prior else num_datasets)
    )
    columns = default_columns if num_cols is None else num_cols
    if not isinstance(columns, (int, np.integer)) or isinstance(columns, bool) or columns <= 0:
        raise ValueError("num_cols must be a positive integer.")
    dataset_indices = draws.data_indices if aggregation is None else [-1]
    panel_specs = []
    if parameter_groups is None:
        for name in names:
            for dataset, index in enumerate(dataset_indices):
                panel_specs.append((len(panel_specs), name, dataset, index))
        rows = int(np.ceil(len(panel_specs) / columns))
    else:
        rows = 0
        for group in parameter_groups:
            for dataset, index in enumerate(dataset_indices):
                for offset, name in enumerate(group):
                    panel_specs.append((rows * columns + offset, name, dataset, index))
                rows += int(np.ceil(len(group) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=figsize or (6 * columns, 3.2 * rows + 0.35),
        squeeze=False,
    )
    display_names = dict(zip(names, display_names))
    legend_color = colors[0]
    for panel, name, column, index in panel_specs:
        scenarios = (draws.parameter_conditions or {}).get(name, draws.conditions)
        labels = [format_condition_label(condition) for condition in scenarios]
        ax = axes.flat[panel]
        if marginal:
            sub = ax.get_subplotspec().subgridspec(1, 2, width_ratios=[4.2, 0.8], wspace=0)
            trajectory_ax = fig.add_subplot(sub[0])
            marginal_ax = fig.add_subplot(sub[1])
            ax.set_visible(False)
        else:
            trajectory_ax, marginal_ax = ax, None
        for condition, label in enumerate(labels):
            values = (
                draws.parameters[name][column, :, condition, :]
                if aggregation is None
                else np.asarray(aggregation(draws.parameters[name][:, :, condition, :], axis=0))
            )
            values = smooth_trajectories(values, smoothing, smoothing_window)
            color = colors[condition]
            steps = np.arange(values.shape[-1])
            individuals = values if prior else values[:num_trajectories]
            for trajectory in individuals:
                trajectory_ax.plot(steps, trajectory, color=color, alpha=alpha, linewidth=1.5)
            if prior:
                center = values.mean(axis=0)
            else:
                center = np.median(values, axis=0) if aggregation is None else np.asarray(aggregation(values, axis=0))
                if uncertainty_fun is not None:
                    outer, inner = compute_uncertainty_bands(values, uncertainty_fun, center)
                    plot_uncertainty_bands(trajectory_ax, steps, outer, inner, color, alpha, center=center)
            trajectory_ax.plot(
                steps,
                center,
                color=color,
                linewidth=2.5,
                label=label,
                marker="o" if len(steps) == 1 else None,
            )
            if marginal_ax is not None:
                plot_dist(
                    values,
                    ax=marginal_ax,
                    dist_type=dist_type,
                    num_bins=num_bins,
                    color=color,
                    alpha=dist_alpha if dist_alpha is not None else (0.45 if condition_count > 1 else 1),
                    orientation="vertical",
                    hide_axis=True,
                )
            if targets is not None:
                target = (
                    targets.parameters[name][column, 0, condition]
                    if aggregation is None
                    else np.asarray(aggregation(targets.parameters[name][:, 0, condition], axis=0))
                )
                target = smooth_trajectories(target, smoothing, smoothing_window)
                trajectory_ax.plot(
                    steps, target, color="black" if condition_count == 1 else color, linestyle="--", linewidth=1.5
                )
                if marginal_ax is not None:
                    plot_dist(
                        target,
                        ax=marginal_ax,
                        dist_type=dist_type,
                        num_bins=num_bins,
                        color="black" if condition_count == 1 else color,
                        orientation="vertical",
                        alpha=0.4,
                        hide_axis=True,
                    )
        title = display_names[name] if prior or index == -1 else f"{display_names[name]} — Dataset {index}"
        trajectory_ax.set_title(title, fontsize=title_fontsize)
        trajectory_ax.set_xlabel("Step" if panel // columns == rows - 1 else "", fontsize=label_fontsize)
        trajectory_ax.set_ylabel("Parameter value" if panel % columns == 0 else "", fontsize=label_fontsize)
        trajectory_ax.tick_params(labelsize=tick_fontsize)
        trajectory_ax.grid(alpha=0.3)
        if marginal_ax is not None:
            marginal_ax.set_ylim(trajectory_ax.get_ylim())

    used_panels = {panel for panel, *_ in panel_specs}
    for panel, ax in enumerate(axes.flat):
        if panel not in used_panels:
            ax.set_visible(False)
    handles = [mlines.Line2D([], [], color=legend_color, linewidth=2.5, label="Average" if prior else "Median")]
    if prior or num_trajectories:
        handles.append(mlines.Line2D([], [], color=legend_color, linewidth=1.5, alpha=alpha, label="Individual draws"))
    if targets is not None:
        handles.append(mlines.Line2D([], [], color="black", linestyle="--", label="Target"))
    if not prior and uncertainty_fun is not None:
        interval_labels = (
            UNCERTAINTY_LABELS[uncertainty_fun] if isinstance(uncertainty_fun, str) else ("Uncertainty", None)
        )
        handles.append(
            mpatches.Patch(facecolor=legend_color, edgecolor="none", alpha=alpha * 0.5, label=interval_labels[0])
        )
        if interval_labels[1] is not None:
            handles.append(
                mpatches.Patch(facecolor=legend_color, edgecolor="none", alpha=alpha, label=interval_labels[1])
            )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        fontsize=min(tick_fontsize, 11),
        frameon=False,
        columnspacing=0.7,
        handlelength=1.3,
        handletextpad=0.5,
    )
    sns.despine(fig=fig)
    fig.tight_layout(rect=(0, 0.3 / fig.get_figheight(), 1, 1))
    return fig
