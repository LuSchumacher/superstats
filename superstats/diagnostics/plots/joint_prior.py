"""Prior sample visualization helpers."""

from collections.abc import Mapping, Sequence
from typing import Literal

import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

from superstats.defaults import (
    BASE_COLOR,
    BASE_ROW_HEIGHT,
    CATEGORICAL_PALETTE,
    JOINT_HSPACE,
    LABEL_FONTSIZE,
    TICK_FONTSIZE,
    TITLE_FONTSIZE,
    WSPACE,
    Y_LABEL_PAD,
)
from superstats.utils.plotting import (
    get_default_num_cols,
    get_layout,
    plot_dist,
    resolve_dist_alpha,
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
    alpha: float = 0.5,
    figsize: tuple[float, float] | None = None,
    condition_labels: Mapping[str, Sequence[str]] | None = None,
    parameter_groups: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
) -> plt.Figure:
    """Plot joint prior diagnostics.

    Combines hyperparameter distributions, shared-parameter distributions,
    and time-varying trajectories. With `parameter_groups`, each observation-
    model parameter has one row: hyperparameters first, then panels titled
    "Raw" and "Implied". Several raw coefficients occupy adjacent panels in
    the same row, with coefficient names below the axes. Stationary quantities
    use distributions in place of trajectories.

    Without grouping metadata, each trajectory retains its own row and shared
    distributions fill subsequent rows.

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
        Font size for subplot titles.
    label_fontsize : int, optional, default: 18
        Font size for row labels and the figure legend.
    tick_fontsize  : int, optional, default: 16
        Font size for tick labels.
    alpha          : float, optional, default: 0.5
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
        both views is rendered once with title "Raw / Implied". All shared
        samples included in groups are rendered in their owning parameter row.
        Only sample keys referenced by the grouping metadata are plotted.

    Returns
    -------
    fig : plt.Figure
        The generated figure.

    Raises
    ------
    ValueError
        If no plottable parameters are found or ``dist_type`` is invalid.
    """

    if parameter_groups is not None:
        row_specs = []
        for name, group in parameter_groups.items():
            raw, implied = list(group.get("raw", ())), list(group.get("implied", ()))
            panels = [(key, "Raw / Implied" if key in implied else "Raw", "raw") for key in raw]
            panels += [(key, "Implied", "implied") for key in implied if key not in raw]
            row_specs.append(
                {
                    "name": name,
                    "hyper_cols": [(key, hyper_params[key]) for key in group.get("hyper", ())],
                    "panels": panels,
                }
            )
        shared_items = []
    else:
        hyper_owners = [name for name, keys in (hyper_param_groups or {}).items() if keys]
        names = list(dict.fromkeys([*local_params, *hyper_owners]))
        row_specs = []
        for name in names:
            keys = (
                hyper_param_groups.get(name, ())
                if hyper_param_groups is not None
                else [key for key in hyper_params if key.startswith(name + "_")]
            )
            row_specs.append(
                {
                    "name": name,
                    "hyper_cols": [(key, hyper_params[key]) for key in keys if key in hyper_params],
                    "panels": [(name, "Trajectory", "implied")] if name in local_params else [],
                }
            )
        shared_items = list(shared_params.items())

    if not row_specs and not shared_items:
        raise ValueError("No plottable parameters found.")
    max_hyper = max((len(row["hyper_cols"]) for row in row_specs), default=0)
    max_raw = max((sum(role == "raw" for _, _, role in row["panels"]) for row in row_specs), default=0)
    max_implied = max((sum(role == "implied" for _, _, role in row["panels"]) for row in row_specs), default=0)
    if parameter_groups is None:
        max_implied = 1
    num_cols = max(max_hyper + max_raw + max_implied, 1)
    widths = [1.0] * max_hyper
    for role, count in (("raw", max_raw), ("implied", max_implied)):
        role_widths = [1.0] * count
        for row in row_specs:
            keys = [key for key, _, panel_role in row["panels"] if panel_role == role]
            for index, key in enumerate(keys):
                if key in local_params:
                    role_widths[index] = 2.0
        widths.extend(role_widths)
    if not widths:
        widths = [1.0]
    shared_num_cols = num_cols + 1 if row_specs else get_default_num_cols(len(shared_items))
    shared_num_rows = int(np.ceil(len(shared_items) / shared_num_cols))
    num_rows = len(row_specs) + shared_num_rows
    plot_figsize, legend_bottom, legend_y = get_layout(
        num_rows,
        num_cols if row_specs else shared_num_cols,
        figsize,
        col_width=4.0,
        row_height=BASE_ROW_HEIGHT + (0.4 if parameter_groups is not None else 0),
    )
    max_label_width = 0.0
    if row_specs:
        label_font = FontProperties(family=plt.rcParams["font.family"], size=label_fontsize)
        max_label_width = (
            max(TextPath((0, 0), row["name"], prop=label_font).get_extents().width for row in row_specs) / 72
        )
        if figsize is None:
            plot_figsize = (plot_figsize[0] + max_label_width + 0.25, plot_figsize[1])
    fig = plt.figure(figsize=plot_figsize)
    label_columns = int(parameter_groups is not None)
    grid_widths = [max_label_width / 2 + 0.15, *widths] if label_columns else widths
    gs = gridspec.GridSpec(num_rows, num_cols + label_columns, width_ratios=grid_widths, figure=fig)

    def plot_distribution(ax, name, values, owner):
        """Draw scalar, conditional, or vector distributions in a prior panel.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis receiving the distributions.
        name : str
            Sample key used to find condition or mixture-component labels.
        values : np.ndarray
            Scalar samples shaped (S,), or grouped samples shaped (S, K).
        owner : str
            Row parameter used as a fallback for mixture-component names.

        Notes
        -----
        Mixture legends are retained; fixed-condition legends are hidden.
        """
        values = np.asarray(values)
        labels = (condition_labels or {}).get(name)
        components = values.shape[1] if values.ndim == 2 else 1
        mixture_owner = name.split("_mixture_weights")[0]
        component_names = (
            labels
            or (mixture_names or {}).get(mixture_owner)
            or (mixture_names or {}).get(owner)
            or [f"component {i}" for i in range(components)]
        )
        for component in range(components):
            component_color = (
                color
                if components == 1
                else CATEGORICAL_PALETTE[component]
                if components <= len(CATEGORICAL_PALETTE)
                else plt.get_cmap("tab20")(component % 20)
            )
            plot_dist(
                values[:, component] if values.ndim == 2 else values.reshape(-1),
                ax=ax,
                dist_type=dist_type,
                num_bins=num_bins,
                color=component_color,
                alpha=resolve_dist_alpha(dist_alpha, components),
                label=component_names[component],
            )
        if components > 1 and not labels:
            ax.legend(fontsize=tick_fontsize, framealpha=0.0)

    for row_index, spec in enumerate(row_specs):
        if label_columns:
            label_ax = fig.add_subplot(gs[row_index, 0])
            label_ax.axis("off")
            label_ax.text(
                1,
                0.5,
                spec["name"],
                transform=label_ax.transAxes,
                ha="right",
                va="center",
                fontsize=label_fontsize,
            )
        axes = [fig.add_subplot(gs[row_index, column + label_columns]) for column in range(num_cols)]
        active = []
        for column, (key, values) in enumerate(spec["hyper_cols"]):
            ax = axes[column]
            plot_distribution(ax, key, values, spec["name"])
            prefix = spec["name"] + "_"
            ax.set_title(key.removeprefix(prefix), fontsize=title_fontsize, pad=15)
            active.append(ax)
        positions = {"raw": max_hyper, "implied": max_hyper + max_raw}
        used = set(range(len(spec["hyper_cols"])))
        for key, title, role in spec["panels"]:
            column = positions[role]
            positions[role] += 1
            used.add(column)
            ax = axes[column]
            if key in local_params:
                marginal_ax = None
                if marginal:
                    sub = ax.get_subplotspec().subgridspec(1, 2, width_ratios=[4.2, 0.8], wspace=0)
                    ax.axis("off")
                    ax = fig.add_subplot(sub[0])
                    marginal_ax = fig.add_subplot(sub[1])
                values = np.asarray(local_params[key])
                groups = values[:, None, :] if values.ndim == 2 else values
                labels = (condition_labels or {}).get(key, [None] * groups.shape[1])
                for group in range(groups.shape[1]):
                    group_color = (
                        color
                        if groups.shape[1] == 1
                        else CATEGORICAL_PALETTE[group]
                        if groups.shape[1] <= len(CATEGORICAL_PALETTE)
                        else plt.get_cmap("tab20")(group % 20)
                    )
                    for trajectory in groups[:, group]:
                        ax.plot(trajectory, alpha=alpha, color=group_color, linewidth=1.5)
                    ax.plot(groups[:, group].mean(axis=0), color=group_color, linewidth=2.5, label=labels[group])
                    if marginal_ax is not None:
                        plot_dist(
                            groups[:, group],
                            ax=marginal_ax,
                            dist_type=dist_type,
                            num_bins=num_bins,
                            color=group_color,
                            orientation="vertical",
                            alpha=resolve_dist_alpha(dist_alpha, groups.shape[1]),
                            hide_axis=True,
                        )
                if param_bounds and key in param_bounds:
                    ax.set_ylim(param_bounds[key])
                if marginal_ax is not None:
                    marginal_ax.set_ylim(ax.get_ylim())
            else:
                plot_distribution(ax, key, shared_params[key], spec["name"])
            ax.set_title(title, fontsize=title_fontsize, pad=15)
            base = key.removesuffix(" [raw]").removesuffix(" [implied]")
            ax.set_xlabel(
                base if parameter_groups is not None and base != spec["name"] else "", fontsize=label_fontsize
            )
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
                fontsize=label_fontsize,
                labelpad=Y_LABEL_PAD,
            )

    for shared_row in range(shared_num_rows):
        sub = gs[len(row_specs) + shared_row, label_columns:].subgridspec(1, shared_num_cols, wspace=2 * WSPACE)
        for column in range(shared_num_cols):
            ax = fig.add_subplot(sub[column])
            index = shared_row * shared_num_cols + column
            if index >= len(shared_items):
                ax.axis("off")
                continue
            name, values = shared_items[index]
            plot_distribution(ax, name, values, name)
            ax.set_title(name, fontsize=title_fontsize, pad=15)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=tick_fontsize)

    fig.legend(
        handles=[
            mlines.Line2D([], [], color=color, linewidth=2.5, label="Average"),
            mlines.Line2D([], [], color=color, linewidth=2.0, label="Individual"),
        ],
        loc="lower center",
        ncol=2,
        fontsize=label_fontsize,
        framealpha=0.0,
        bbox_to_anchor=(0.5, legend_y),
    )
    fig.tight_layout()
    fig.subplots_adjust(
        bottom=legend_bottom,
        hspace=JOINT_HSPACE + (0.2 if parameter_groups is not None else 0),
        wspace=WSPACE,
    )
    sns.despine()
    return fig
