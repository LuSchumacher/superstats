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
    BASE_COL_WIDTH,
    BASE_ROW_HEIGHT,
    CATEGORICAL_PALETTE,
    LABEL_PAD,
    Y_LABEL_PAD,
)
from superstats.utils.plotting import get_default_num_cols, get_layout, plot_dist


plt.rcParams["axes.axisbelow"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = [
    "Palatino",
    "Palatino Linotype",
    "DejaVu Serif",
]


def plot_time_varying_prior(
    local_params: Mapping[str, np.ndarray],
    param_bounds: Mapping[str, tuple[float, float]] | None = None,
    num_cols: int | None = None,
    marginal: bool = True,
    dist_type: Literal["hist", "kde", "both"] = "hist",
    num_bins: int | None = 40,
    alpha: float = 0.5,
    color: str = BASE_COLOR,
    title_fontsize: int = 22,
    label_fontsize: int = 18,
    tick_fontsize: int = 16,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Plot time-varying parameter trajectories.

    Individual trajectories and their average are shown for every parameter.
    An optional marginal histogram or KDE is displayed beside each trajectory
    panel.

    Parameters
    ----------
    local_params : dict of np.ndarray
        Mapping from parameter names to arrays with shape
        ``(num_trajectories, num_steps)``.
    param_bounds : dict or None, optional, default: None
        Optional mapping from parameter names to ``(lower, upper)`` y-axis
        limits.
    num_cols : int or None, optional, default: None
        Number of subplot columns. If ``None``, uses the compact dynamic
        layout shared by the distribution plots.
    marginal : bool, optional, default: True
        Whether to display a marginal panel.
    dist_type : {"hist", "kde", "both"}, optional, default: "hist"
        Marginal plot type: ``"hist"``, ``"kde"``, or ``"both"``.
    num_bins : int or None, optional, default: 40
        Number of histogram bins. If None, Seaborn selects the bins.
    alpha : float, optional, default: 0.5
        Opacity of individual trajectories.
    color : str, optional, default: BASE_COLOR
        Color used for trajectories and marginals.
    title_fontsize : int, optional, default: 22
        Font size for subplot titles.
    label_fontsize : int, optional, default: 18
        Font size for axis labels and the legend.
    tick_fontsize : int, optional, default: 16
        Font size for tick labels.
    figsize : tuple of two floats or None, optional, default: None
        Optional figure size in inches.

    Returns
    -------
    fig : plt.Figure
        The generated figure.

    Raises
    ------
    ValueError
        If ``dist_type`` is invalid.
    """

    n = len(local_params)
    if num_cols is None:
        num_cols = get_default_num_cols(n)
    num_rows = int(np.ceil(n / num_cols))

    plot_figsize, legend_bottom, legend_y = get_layout(
        num_rows,
        num_cols,
        figsize,
        col_width=BASE_COL_WIDTH,
        row_height=BASE_ROW_HEIGHT,
    )

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=plot_figsize,
    )
    axes = np.atleast_1d(axes).ravel()

    for i, (name, values) in enumerate(local_params.items()):
        ax = axes[i]
        values_plot = np.asarray(values)

        if marginal:
            sub = ax.get_subplotspec().subgridspec(
                1,
                2,
                width_ratios=[4.2, 0.8],
                wspace=0.0,
            )
            ax_traj = fig.add_subplot(sub[0])
            ax_marginal = fig.add_subplot(sub[1])
            ax.axis("off")
        else:
            ax_traj = ax
            ax_marginal = None

        if param_bounds and name in param_bounds:
            ax_traj.set_ylim(param_bounds[name])

        for trajectory in values_plot:
            ax_traj.plot(
                trajectory,
                alpha=alpha,
                color=color,
                linewidth=1.5,
            )

        ax_traj.plot(
            values_plot.mean(axis=0),
            color=color,
            linewidth=2.5,
            alpha=1.0,
        )

        ax_traj.set_title(
            name,
            fontsize=title_fontsize,
            pad=15,
        )
        ax_traj.set_xlabel("")
        ax_traj.set_ylabel("")

        if i // num_cols == num_rows - 1:
            ax_traj.set_xlabel(
                "Step",
                fontsize=label_fontsize,
                labelpad=LABEL_PAD,
            )

        if i % num_cols == 0:
            ax_traj.set_ylabel(
                "Value",
                fontsize=label_fontsize,
                labelpad=Y_LABEL_PAD,
            )

        ax_traj.grid(alpha=0.3)
        ax_traj.tick_params(labelsize=tick_fontsize)

        if ax_marginal is not None:
            plot_dist(
                values_plot,
                ax=ax_marginal,
                dist_type=dist_type,
                color=color,
                orientation="vertical",
                num_bins=num_bins,
                hide_axis=True,
            )
            ax_marginal.set_ylim(ax_traj.get_ylim())

    for j in range(len(local_params), len(axes)):
        axes[j].axis("off")

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
                linewidth=1.0,
                label="Individual",
            ),
        ],
        loc="lower center",
        ncol=2,
        fontsize=label_fontsize,
        framealpha=0.0,
        bbox_to_anchor=(0.5, legend_y),
    )

    sns.despine()
    plt.tight_layout()
    fig.subplots_adjust(bottom=legend_bottom)

    return fig


def plot_time_invariant_prior(
    hyper_params: Mapping[str, np.ndarray],
    shared_params: Mapping[str, np.ndarray],
    mixture_names: Mapping[str, Sequence[str]] | None = None,
    dist_type: Literal["hist", "kde", "both"] = "hist",
    num_bins: int | None = 40,
    color: str = BASE_COLOR,
    num_cols: int | None = None,
    title_fontsize: int = 22,
    label_fontsize: int = 18,
    tick_fontsize: int = 16,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Plot time-invariant hyperparameter and shared-parameter distributions.

    Two-dimensional arrays with multiple columns are treated as mixture
    components and plotted separately.

    Parameters
    ----------
    hyper_params : dict of np.ndarray
        Mapping from hyperparameter names to sample arrays.
    shared_params : dict of np.ndarray
        Mapping from shared parameter names to sample arrays.
    mixture_names : dict or None, optional, default: None
        Optional mapping from parameter names to mixture-component labels.
    dist_type : {"hist", "kde", "both"}, optional, default: "both"
        Distribution plot type.
    num_bins : int or None, optional, default: 40
        Number of histogram bins. If None, Seaborn selects the bins.
    color : str, optional, default: BASE_COLOR
        Color for non-mixture distributions.
    num_cols : int or None, optional, default: None
        Number of subplot columns. If ``None``, uses a compact default layout:
        1--3 parameters use 1--3 columns, 4 uses 2, 5--6 use 3, 7--8 use
        4, 9 uses 3, and 10 or more use 4.
    title_fontsize : int, optional, default: 22
        Font size for subplot titles.
    label_fontsize : int, optional, default: 18
        Font size for axis labels.
    tick_fontsize : int, optional, default: 16
        Font size for tick labels and legends.
    figsize : tuple of two floats or None, optional, default: None
        Optional figure size in inches.

    Returns
    -------
    fig : plt.Figure
        The generated figure.

    Raises
    ------
    ValueError
        If both parameter mappings are empty.
    """

    labeled_params = {
        **{f"{name}  [hyper]": values for name, values in hyper_params.items()},
        **{f"{name}  [shared]": values for name, values in shared_params.items()},
    }

    n = len(labeled_params)
    if num_cols is None:
        num_cols = get_default_num_cols(n)

    num_rows = int(np.ceil(n / num_cols))

    plot_figsize, legend_bottom, legend_y = get_layout(
        num_rows,
        num_cols,
        figsize,
        col_width=BASE_COL_WIDTH,
        row_height=BASE_ROW_HEIGHT,
    )

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=plot_figsize,
    )
    axes = np.atleast_1d(axes).ravel()

    for i, (label, values) in enumerate(labeled_params.items()):
        ax = axes[i]
        arr = np.asarray(values)

        if arr.ndim == 2 and arr.shape[1] > 1:
            param_name = label.split("_mixture_weights")[0].strip()

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
                    label=component_names[k],
                )

            ax.legend(
                fontsize=tick_fontsize,
                framealpha=0.0,
            )
        else:
            plot_dist(
                arr.reshape(-1),
                ax=ax,
                dist_type=dist_type,
                color=color,
                num_bins=num_bins,
            )

        ax.set_title(
            label,
            fontsize=title_fontsize,
            pad=10,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")

        if i // num_cols == num_rows - 1:
            ax.set_xlabel(
                "Value",
                fontsize=label_fontsize,
                labelpad=LABEL_PAD,
            )

        if i % num_cols == 0:
            ax.set_ylabel(
                "Count" if dist_type == "hist" else "Density",
                fontsize=label_fontsize,
                labelpad=Y_LABEL_PAD,
            )

        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=tick_fontsize)

    for j in range(len(labeled_params), len(axes)):
        axes[j].axis("off")

    sns.despine()
    plt.tight_layout()
    fig.subplots_adjust(bottom=legend_bottom)

    return fig


def plot_joint_prior(
    local_params: Mapping[str, np.ndarray],
    hyper_params: Mapping[str, np.ndarray],
    shared_params: Mapping[str, np.ndarray],
    param_bounds: Mapping[str, tuple[float, float]] | None = None,
    mixture_names: Mapping[str, Sequence[str]] | None = None,
    hyper_param_groups: Mapping[str, Sequence[str]] | None = None,
    marginal: bool = True,
    dist_type: Literal["hist", "kde", "both"] = "hist",
    num_bins: int | None = 40,
    color: str = BASE_COLOR,
    title_fontsize: int = 22,
    tick_fontsize: int = 16,
    alpha: float = 0.5,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Plot joint prior diagnostics.

    Combines hyperparameter distributions, shared-parameter distributions,
    and time-varying trajectories. Each row corresponds to one parameter.

    Parameters
    ----------
    local_params : dict of np.ndarray
        Mapping from parameter names to trajectory arrays with shape
        ``(num_trajectories, num_steps)``.
    hyper_params : dict of np.ndarray
        Mapping from hyperparameter names to sample arrays.
    shared_params : dict of np.ndarray
        Mapping from parameter names to shared-parameter sample arrays.
    param_bounds : dict or None, optional, default: None
        Optional mapping from parameter names to trajectory y-axis limits.
    mixture_names : dict or None, optional, default: None
        Optional mapping from parameter names to mixture-component labels.
    hyper_param_groups : dict or None, optional, default: None
        Optional mapping from parameter names to the exact hyperparameter
        keys associated with them.
    marginal : bool, optional, default: True
        Whether to display a marginal trajectory panel.
    dist_type : {"hist", "kde", "both"}, optional, default: "hist"
        Marginal plot type: ``"hist"``, ``"kde"``, or ``"both"``.
    num_bins : int or None, optional, default: 40
        Number of histogram bins. If None, Seaborn selects the bins.
    color : str, optional, default: BASE_COLOR
        Color used for trajectories and distributions.
    title_fontsize : int, optional, default: 22
        Font size for subplot titles.
    tick_fontsize : int, optional, default: 16
        Font size for tick labels.
    alpha : float, optional, default: 0.5
        Opacity of individual trajectories.
    figsize : tuple of two floats or None, optional, default: None
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

    all_param_names = list(dict.fromkeys(list(local_params.keys()) + list(shared_params.keys())))

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
                "shared": (np.asarray(shared_params[param_name]) if param_name in shared_params else None),
            }
        )

    max_hyper = max(len(row["hyper_cols"]) for row in row_specs)
    num_cols = max_hyper + 1
    num_rows = len(row_specs)

    plot_figsize, legend_bottom, legend_y = get_layout(
        num_rows,
        num_cols,
        figsize,
        col_width=4.0,
        row_height=BASE_ROW_HEIGHT,
        legend_offset=0.25,
    )

    if figsize is None:
        label_font = FontProperties(
            family=plt.rcParams["font.family"],
            size=title_fontsize,
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

    axes = np.array([[fig.add_subplot(gs[row_i, col_i]) for col_i in range(num_cols)] for row_i in range(num_rows)])

    for row_i, spec in enumerate(row_specs):
        param_name = spec["name"]
        hyper_cols = spec["hyper_cols"]
        local_arr = spec["local"]
        shared_arr = spec["shared"]
        prefix = param_name + "_"

        for col_i, (label, values) in enumerate(hyper_cols):
            ax = axes[row_i, col_i]
            arr = np.asarray(values)

            if arr.ndim == 2 and arr.shape[1] > 1:
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
                        label=component_names[k],
                    )

                ax.legend(
                    fontsize=tick_fontsize,
                    framealpha=0.0,
                )
            else:
                plot_dist(
                    arr.reshape(-1),
                    ax=ax,
                    dist_type=dist_type,
                    color=color,
                    num_bins=num_bins,
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

        if shared_arr is not None:
            ax = axes[row_i, 0]

            plot_dist(
                shared_arr.reshape(-1),
                ax=ax,
                dist_type=dist_type,
                color=color,
                num_bins=num_bins,
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
                plot_dist(
                    local_arr,
                    ax=ax_marginal,
                    dist_type=dist_type,
                    color=color,
                    orientation="vertical",
                    num_bins=num_bins,
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
            if shared_arr is None or col_i > 0:
                axes[row_i, col_i].axis("off")

        row_label_ax = axes[row_i, 0] if hyper_cols or shared_arr is not None else ax_traj
        row_label_ax.set_ylabel(
            param_name,
            rotation=0,
            ha="right",
            va="center",
            fontsize=title_fontsize,
            labelpad=Y_LABEL_PAD,
        )

    fig.legend(
        handles=[
            mlines.Line2D(
                [],
                [],
                color="black",
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
        fontsize=title_fontsize - 2,
        framealpha=0.0,
        bbox_to_anchor=(0.5, legend_y),
    )

    fig.tight_layout()
    fig.subplots_adjust(
        bottom=legend_bottom,
        hspace=0.5,
    )

    sns.despine()

    return fig
