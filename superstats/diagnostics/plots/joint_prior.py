"""Prior sample visualization helpers."""

from collections.abc import Mapping, Sequence
from typing import Literal

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

from superstats.defaults import (
    BASE_COLOR,
    LABEL_FONTSIZE,
    TICK_FONTSIZE,
    TITLE_FONTSIZE,
    Y_LABEL_PAD,
)
from superstats.utils.plotting import (
    PRIOR_ALPHA,
    PRIOR_ROW_HEIGHT,
    PRIOR_SECTION_ROW_HEIGHT,
    PRIOR_TITLE_PAD,
    add_prior_heading,
    add_prior_trajectory_legend,
    adjust_prior_layout,
    get_default_num_cols,
    get_layout,
    plot_prior_distribution,
    plot_prior_trajectories,
)


def plot_joint_prior(
    local_params: Mapping[str, np.ndarray],
    hyper_params: Mapping[str, np.ndarray],
    shared_params: Mapping[str, np.ndarray],
    param_bounds: Mapping[str, tuple[float, float]] | None = None,
    mixture_names: Mapping[str, Sequence[str]] | None = None,
    hyper_param_groups: Mapping[str, Sequence[str]] | None = None,
    marginal: bool = True,
    dist_type: Literal["hist", "kde", "both"] = "hist",
    num_bins: int | None = None,
    dist_alpha: float | None = None,
    color: str = BASE_COLOR,
    title_fontsize: int = TITLE_FONTSIZE,
    label_fontsize: int = LABEL_FONTSIZE,
    tick_fontsize: int = TICK_FONTSIZE,
    alpha: float = PRIOR_ALPHA,
    figsize: tuple[float, float] | None = None,
    condition_labels: Mapping[str, Sequence[str]] | None = None,
    parameter_groups: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
) -> plt.Figure:
    """Plot joint prior diagnostics.

    Combines hyperparameter distributions, shared-parameter distributions,
    and time-varying trajectories. With `parameter_groups`, trajectory rows
    are labeled by observation-model parameter, with hyperparameters on the
    left. Several raw coefficients occupy adjacent panels, with coefficient
    names below the axes. Shared distributions appear in a dedicated section
    aligned with the hyperparameter columns.

    Parameters
    ----------
    local_params   : dict of np.ndarray
        Mapping from parameter names to trajectory arrays with shape
        (num_trajectories, num_steps), or
        (num_trajectories, num_conditions, num_steps).
    hyper_params   : dict of np.ndarray
        Mapping from hyperparameter names to sample arrays.
    shared_params  : dict of np.ndarray
        Mapping from parameter names to shared-parameter sample arrays.
    param_bounds   : dict or None, optional, default: None
        Optional mapping from parameter names to trajectory y-axis limits.
    mixture_names  : dict or None, optional, default: None
        Optional mapping from parameter names to mixture-component labels.
    hyper_param_groups : dict or None, optional, default: None
        Optional mapping from parameter names to the exact hyperparameter
        keys associated with them.
    marginal       : bool, optional, default: True
        Whether to display a marginal trajectory panel.
    dist_type      : {"hist", "kde", "both"}, optional, default: "hist"
        Marginal plot type: ``"hist"``, ``"kde"``, or ``"both"``.
    num_bins       : int or None, optional, default: None
        Number of histogram bins. If None, Seaborn selects the bins.
    dist_alpha     : float or None, optional, default: None
        Opacity of all marginal and time-invariant distributions. If None,
        uses 1.0 for one distribution and 0.5 for overlaid mixture components.
    color          : str, optional, default: BASE_COLOR
        Color used for trajectories and distributions.
    title_fontsize : int, optional, default: 22
        Font size for subplot titles, section headings, and row labels.
    label_fontsize : int, optional, default: 18
        Font size for axis labels, scale headings, and the figure legend.
    tick_fontsize  : int, optional, default: 16
        Font size for tick labels.
    alpha          : float, optional, default: 0.25
        Opacity of individual trajectories.
    figsize        : tuple of two floats or None, optional, default: None
        Optional figure size in inches.
    condition_labels : mapping of str to sequence of str or None, optional, default: None
        Labels identifying conditional trajectory and distribution groups.
        Condition legends are hidden.
    parameter_groups : mapping or None, optional, default: None
        Observation-parameter row names mapped to dictionaries with "hyper",
        "raw", and "implied" sample-key lists. Hyper keys refer to `hyper_params`;
        view keys refer to `local_params` or `shared_params`. A key included in
        both views is rendered once. Only sample keys referenced by the
        grouping metadata are plotted.

    Returns
    -------
    fig : plt.Figure
        The generated figure.

    Raises
    ------
    ValueError
        If no plottable parameters are found or ``dist_type`` is invalid.
    """

    row_specs = []
    shared_owners = {}
    if parameter_groups is not None:
        for name, group in parameter_groups.items():
            raw, implied = list(group.get("raw", ())), list(group.get("implied", ()))
            for key in dict.fromkeys([*raw, *implied]):
                if key in shared_params and key not in local_params:
                    shared_owners.setdefault(key, name)
            panels = [(key, "raw", key in implied) for key in raw if key in local_params]
            panels += [(key, "implied", False) for key in implied if key in local_params and key not in raw]
            hyper_cols = [(key, hyper_params[key]) for key in group.get("hyper", ())]
            if panels or hyper_cols:
                row_specs.append({"name": name, "hyper_cols": hyper_cols, "panels": panels})
    else:
        hyper_owners = [name for name, keys in (hyper_param_groups or {}).items() if keys]
        names = list(dict.fromkeys([*local_params, *hyper_owners]))
        for name in names:
            keys = (
                hyper_param_groups.get(name, ())
                if hyper_param_groups is not None
                else [key for key in hyper_params if key.startswith(name + "_")]
            )
            hyper_cols = [(key, hyper_params[key]) for key in keys if key in hyper_params]
            panels = [(name, "implied", False)] if name in local_params else []
            if panels or hyper_cols:
                row_specs.append({"name": name, "hyper_cols": hyper_cols, "panels": panels})
        shared_owners = {key: key for key in shared_params}

    shared_items = list(shared_owners.items())
    if not row_specs and not shared_items:
        raise ValueError("No plottable parameters found.")
    max_hyper = max((len(row["hyper_cols"]) for row in row_specs), default=0)
    max_raw = max((sum(role == "raw" for _, role, _ in row["panels"]) for row in row_specs), default=0)
    max_implied = max((sum(role == "implied" for _, role, _ in row["panels"]) for row in row_specs), default=0)
    has_trajectories = max_raw + max_implied > 0
    show_scales = max_raw > 0 and max_implied > 0
    dist_cols = max_hyper or (get_default_num_cols(len(shared_items)) if shared_items else 0)
    num_cols = dist_cols + max_raw + max_implied
    widths = [1.0] * dist_cols + [2.0] * (max_raw + max_implied)
    shared_num_rows = int(np.ceil(len(shared_items) / dist_cols)) if shared_items else 0
    shared_gap = bool(row_specs and shared_items)
    shared_start = len(row_specs) + shared_gap
    num_plot_rows = len(row_specs) + shared_num_rows
    num_rows = num_plot_rows + shared_gap
    heading_offset = max(title_fontsize, label_fontsize) + PRIOR_TITLE_PAD + 10
    heading_height = (heading_offset + title_fontsize + 10) / 72
    row_height = PRIOR_SECTION_ROW_HEIGHT if shared_gap else PRIOR_ROW_HEIGHT
    plot_figsize, legend_bottom, legend_y = get_layout(
        num_plot_rows,
        num_cols,
        figsize,
        col_width=4.0,
        row_height=row_height + (heading_height + 0.65 * shared_gap) / num_plot_rows,
        legend_space=1.25 if has_trajectories else 0.55,
    )
    label_width = 0.0
    if row_specs:
        label_font = FontProperties(family=plt.rcParams["font.family"], size=title_fontsize)
        max_label_width = (
            max(TextPath((0, 0), row["name"], prop=label_font).get_extents().width for row in row_specs) / 72
        )
        label_width = max_label_width + (0.65 if parameter_groups is not None else 0.25)
    label_columns = int(parameter_groups is not None and bool(row_specs))
    if figsize is None:
        plot_figsize = (plot_figsize[0] + label_width, plot_figsize[1])
    fig = plt.figure(figsize=plot_figsize)
    label_ratio = label_width * sum(widths) / max(plot_figsize[0] - label_width, 1)
    grid_widths = [label_ratio, *widths] if label_columns else widths
    row_heights = [1] * num_rows
    if shared_gap:
        row_heights[shared_start - 1] = 0
    gs = gridspec.GridSpec(
        num_rows, num_cols + label_columns, width_ratios=grid_widths, height_ratios=row_heights, figure=fig
    )

    def plot_distribution(ax, name, values, owner):
        mixture_owner = name.split("_mixture_weights")[0]
        plot_prior_distribution(
            ax,
            values,
            labels=(condition_labels or {}).get(name),
            component_names=(mixture_names or {}).get(mixture_owner) or (mixture_names or {}).get(owner),
            dist_type=dist_type,
            num_bins=num_bins,
            dist_alpha=dist_alpha,
            color=color,
            fontsize=tick_fontsize,
        )

    for row_index, spec in enumerate(row_specs):
        if label_columns:
            label_ax = fig.add_subplot(gs[row_index, 0])
            label_ax.axis("off")
            label_ax.text(
                0,
                0.5,
                spec["name"],
                transform=label_ax.transAxes,
                ha="left",
                va="center",
                fontsize=title_fontsize,
            )
        axes = [fig.add_subplot(gs[row_index, column + label_columns]) for column in range(num_cols)]
        active = []
        for column, (key, values) in enumerate(spec["hyper_cols"]):
            ax = axes[column]
            plot_distribution(ax, key, values, spec["name"])
            prefix = spec["name"] + "_"
            ax.set_title(key.removeprefix(prefix), fontsize=title_fontsize, pad=PRIOR_TITLE_PAD)
            active.append(ax)
        positions = {"raw": dist_cols, "implied": dist_cols + max_raw}
        used = set(range(len(spec["hyper_cols"])))
        for key, role, identical_views in spec["panels"]:
            column = positions[role]
            positions[role] += 1
            used.add(column)
            ax = plot_prior_trajectories(
                axes[column],
                local_params[key],
                bounds=(param_bounds or {}).get(key),
                labels=(condition_labels or {}).get(key),
                marginal=marginal,
                dist_type=dist_type,
                num_bins=num_bins,
                dist_alpha=dist_alpha,
                alpha=alpha,
                color=color,
            )
            base = key.removesuffix(" [raw]").removesuffix(" [implied]")
            panel_label = base if base != spec["name"] else ""
            if show_scales and identical_views:
                panel_label = f"{base} (raw and implied)"
            ax.set_xlabel(panel_label, fontsize=label_fontsize)
            active.append(ax)
        for column, ax in enumerate(axes):
            if column not in used:
                ax.axis("off")
        for ax in active:
            ax.set_ylabel("")
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=tick_fontsize)
        if active and not label_columns:
            active[0].set_ylabel(
                spec["name"],
                rotation=0,
                ha="right",
                va="center",
                fontsize=title_fontsize,
                labelpad=Y_LABEL_PAD,
            )

    for index, (name, owner) in enumerate(shared_items):
        row, column = divmod(index, dist_cols)
        ax = fig.add_subplot(gs[shared_start + row, label_columns + column])
        plot_distribution(ax, name, shared_params[name], owner)
        ax.set_title(name, fontsize=title_fontsize, pad=PRIOR_TITLE_PAD)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=tick_fontsize)

    if has_trajectories:
        add_prior_trajectory_legend(fig, color, alpha, label_fontsize, legend_y)
    adjust_prior_layout(
        fig,
        gs,
        heading_height=heading_height,
        bottom=legend_bottom,
        min_gap=max(title_fontsize + PRIOR_TITLE_PAD, label_fontsize) + tick_fontsize + 13,
        section_gap_row=shared_start - 1 if shared_gap else None,
        section_gap_height=heading_offset + title_fontsize + tick_fontsize + 20,
    )

    raw_start = label_columns + dist_cols
    implied_start = raw_start + max_raw
    if max_hyper:
        add_prior_heading(fig, gs[0, label_columns:raw_start], "Hyperparameters", heading_offset, title_fontsize)
    if has_trajectories:
        add_prior_heading(fig, gs[0, raw_start:], "Trajectories", heading_offset, title_fontsize)
    if show_scales:
        add_prior_heading(fig, gs[0, raw_start:implied_start], "Raw", PRIOR_TITLE_PAD, label_fontsize, "normal")
        add_prior_heading(fig, gs[0, implied_start:], "Implied", PRIOR_TITLE_PAD, label_fontsize, "normal")
    if shared_items:
        add_prior_heading(
            fig, gs[shared_start, label_columns:raw_start], "Shared parameters", heading_offset, title_fontsize
        )
    sns.despine(fig=fig)
    return fig
