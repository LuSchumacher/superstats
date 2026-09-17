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
) -> plt.Figure:
    """Plot joint prior diagnostics.

    Combines hyperparameter distributions, shared-parameter distributions,
    and time-varying trajectories. Time-varying parameters each have a
    dedicated row; shared distributions fill equal-width panels across
    subsequent rows, with parameter names as panel titles.

    Parameters
    ----------
    local_params   : dict of np.ndarray
        Mapping from parameter names to trajectory arrays with shape
        ``(num_trajectories, num_steps)``.
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

    Returns
    -------
    fig : plt.Figure
        The generated figure.

    Raises
    ------
    ValueError
        If no plottable parameters are found or ``dist_type`` is invalid.
    """

    hyper_owners = [name for name, keys in (hyper_param_groups or {}).items() if keys]
    all_param_names = list(dict.fromkeys(list(local_params.keys()) + hyper_owners))

    row_specs = []

    for param_name in all_param_names:
        if hyper_param_groups is not None:
            owned_keys = hyper_param_groups.get(param_name, [])
            hyper_cols = [(key, np.asarray(hyper_params[key])) for key in owned_keys if key in hyper_params]
        else:
            hyper_cols = [
                (key, np.asarray(value)) for key, value in hyper_params.items() if key.startswith(param_name + "_")
            ]

        row_specs.append(
            {
                "name": param_name,
                "hyper_cols": hyper_cols,
                "local": (np.asarray(local_params[param_name]) if param_name in local_params else None),
            }
        )

    if not row_specs and not shared_params:
        raise ValueError("No plottable parameters found.")
    max_hyper = max((len(row["hyper_cols"]) for row in row_specs), default=0)
    num_cols = max_hyper + 1
    # A trajectory cell has twice the width of a distribution cell.
    shared_num_cols = num_cols + 1 if row_specs else get_default_num_cols(len(shared_params))
    shared_num_rows = int(np.ceil(len(shared_params) / shared_num_cols))
    num_rows = len(row_specs) + shared_num_rows

    plot_figsize, legend_bottom, legend_y = get_layout(
        num_rows,
        num_cols if row_specs else shared_num_cols,
        figsize,
        col_width=4.0,
        row_height=BASE_ROW_HEIGHT,
    )

    if figsize is None and row_specs:
        label_font = FontProperties(
            family=plt.rcParams["font.family"],
            size=label_fontsize,
        )
        max_label_width = (
            max(
                TextPath(
                    (0.0, 0.0),
                    row["name"],
                    prop=label_font,
                )
                .get_extents()
                .width
                for row in row_specs
            )
            / 72.0
        )
        plot_figsize = (
            plot_figsize[0] + max_label_width + 0.25,
            plot_figsize[1],
        )

    fig = plt.figure(figsize=plot_figsize)

    gs = gridspec.GridSpec(
        num_rows,
        num_cols,
        width_ratios=[1.0] * (num_cols - 1) + [2.0],
        figure=fig,
    )

    axes = np.array(
        [[fig.add_subplot(gs[row_i, col_i]) for col_i in range(num_cols)] for row_i in range(len(row_specs))]
    )

    for row_i, spec in enumerate(row_specs):
        param_name = spec["name"]
        hyper_cols = spec["hyper_cols"]
        local_arr = spec["local"]
        prefix = param_name + "_"

        for col_i, (label, values) in enumerate(hyper_cols):
            ax = axes[row_i, col_i]
            arr = np.asarray(values)

            if arr.ndim == 2 and arr.shape[1] > 1:
                panel_dist_alpha = resolve_dist_alpha(dist_alpha, arr.shape[1])
                component_names = (mixture_names.get(param_name) if mixture_names else None) or [
                    f"component {k}" for k in range(arr.shape[1])
                ]

                for k in range(arr.shape[1]):
                    plot_dist(
                        arr[:, k],
                        ax=ax,
                        dist_type=dist_type,
                        color=CATEGORICAL_PALETTE[k % len(CATEGORICAL_PALETTE)],
                        num_bins=num_bins,
                        alpha=panel_dist_alpha,
                        label=component_names[k],
                    )

                ax.legend(
                    fontsize=tick_fontsize,
                    framealpha=0.0,
                )
            else:
                panel_dist_alpha = resolve_dist_alpha(dist_alpha, 1)
                plot_dist(
                    arr.reshape(-1),
                    ax=ax,
                    dist_type=dist_type,
                    color=color,
                    num_bins=num_bins,
                    alpha=panel_dist_alpha,
                )

            short_label = label[len(prefix) :] if label.startswith(prefix) else label

            ax.set_title(
                short_label,
                fontsize=title_fontsize,
                pad=15,
            )
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=tick_fontsize)

        ax_cell = axes[row_i, num_cols - 1]

        if marginal:
            sub = ax_cell.get_subplotspec().subgridspec(
                1,
                2,
                width_ratios=[4.2, 0.8],
                wspace=0.0,
            )
            ax_traj = fig.add_subplot(sub[0])
            ax_marginal = fig.add_subplot(sub[1])
            ax_cell.axis("off")
        else:
            ax_traj = ax_cell
            ax_marginal = None

        if local_arr is not None:
            for trajectory in local_arr:
                ax_traj.plot(
                    trajectory,
                    alpha=alpha,
                    color=color,
                    linewidth=1.5,
                )

            ax_traj.plot(
                local_arr.mean(axis=0),
                color=color,
                linewidth=2.5,
                alpha=1.0,
            )

            if param_bounds and param_name in param_bounds:
                ax_traj.set_ylim(param_bounds[param_name])

            ax_traj.set_title(
                "Trajectory",
                fontsize=title_fontsize,
                pad=15,
            )
            ax_traj.set_xlabel("")
            ax_traj.grid(alpha=0.3)
            ax_traj.tick_params(labelsize=tick_fontsize)

            if ax_marginal is not None:
                panel_dist_alpha = resolve_dist_alpha(dist_alpha, 1)
                plot_dist(
                    local_arr,
                    ax=ax_marginal,
                    dist_type=dist_type,
                    color=color,
                    orientation="vertical",
                    num_bins=num_bins,
                    alpha=panel_dist_alpha,
                    hide_axis=True,
                )
                ax_marginal.set_ylim(ax_traj.get_ylim())
        else:
            ax_traj.axis("off")

            if ax_marginal is not None:
                ax_marginal.axis("off")

        for col_i in range(
            len(hyper_cols),
            num_cols - 1,
        ):
            axes[row_i, col_i].axis("off")

        row_label_ax = axes[row_i, 0] if hyper_cols else ax_traj
        row_label_ax.set_ylabel(
            param_name,
            rotation=0,
            ha="right",
            va="center",
            fontsize=label_fontsize,
            labelpad=Y_LABEL_PAD,
        )

    shared_items = list(shared_params.items())
    for shared_row in range(shared_num_rows):
        sub = gs[len(row_specs) + shared_row, :].subgridspec(1, shared_num_cols, wspace=WSPACE)
        for col_i in range(shared_num_cols):
            ax = fig.add_subplot(sub[col_i])
            index = shared_row * shared_num_cols + col_i
            if index >= len(shared_items):
                ax.axis("off")
                continue
            name, values = shared_items[index]
            plot_dist(
                np.asarray(values).reshape(-1),
                ax=ax,
                dist_type=dist_type,
                color=color,
                num_bins=num_bins,
                alpha=resolve_dist_alpha(dist_alpha, 1),
            )
            ax.set_title(name, fontsize=title_fontsize, pad=15)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=tick_fontsize)

    fig.legend(
        handles=[
            mlines.Line2D(
                [],
                [],
                color=color,
                linewidth=2.5,
                label="Average",
            ),
            mlines.Line2D(
                [],
                [],
                color=color,
                linewidth=2.0,
                label="Individual",
            ),
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
        hspace=JOINT_HSPACE,
        wspace=WSPACE,
    )

    sns.despine()

    return fig
