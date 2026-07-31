"""Prior sample visualization helpers."""

from collections.abc import Mapping
from typing import Literal

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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
from superstats.utils.plotting import (
    get_default_num_cols,
    get_layout,
    plot_dist,
)


def plot_time_varying_prior(
    local_params: Mapping[str, np.ndarray],
    param_bounds: Mapping[str, tuple[float, float]] | None = None,
    num_cols: int | None = None,
    marginal: bool = True,
    dist_type: Literal["hist", "kde", "both"] = "hist",
    num_bins: int | None = None,
    dist_alpha: float = 1.0,
    alpha: float = 0.5,
    color: str = BASE_COLOR,
    title_fontsize: int = TITLE_FONTSIZE,
    label_fontsize: int = LABEL_FONTSIZE,
    tick_fontsize: int = TICK_FONTSIZE,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Plot time-varying parameter trajectories.

    Individual trajectories and their average are shown for every parameter.
    An optional marginal histogram or KDE is displayed beside each trajectory
    panel.

    Parameters
    ----------
    local_params   : dict of np.ndarray
        Mapping from parameter names to arrays with shape
        ``(num_trajectories, num_steps)``.
    param_bounds   : dict or None, optional, default: None
        Optional mapping from parameter names to ``(lower, upper)`` y-axis
        limits.
    num_cols       : int or None, optional, default: None
        Number of subplot columns. If ``None``, uses the compact dynamic
        layout shared by the distribution plots.
    marginal       : bool, optional, default: True
        Whether to display a marginal panel.
    dist_type      : {"hist", "kde", "both"}, optional, default: "hist"
        Marginal plot type: ``"hist"``, ``"kde"``, or ``"both"``.
    num_bins       : int or None, optional, default: None
        Number of histogram bins. If None, Seaborn selects the bins.
    dist_alpha     : float, optional, default: 1.0
        Opacity of marginal distributions.
    alpha          : float, optional, default: 0.5
        Opacity of individual trajectories.
    color          : str, optional, default: BASE_COLOR
        Color used for trajectories and marginals.
    title_fontsize : int, optional, default: 22
        Font size for subplot titles.
    label_fontsize : int, optional, default: 18
        Font size for axis labels and the legend.
    tick_fontsize  : int, optional, default: 16
        Font size for tick labels.
    figsize        : tuple of two floats or None, optional, default: None
        Optional figure size in inches.

    Returns
    -------
    fig : plt.Figure
        The generated figure.
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
                "Parameter value",
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
                alpha=dist_alpha,
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
    fig.subplots_adjust(
        bottom=legend_bottom,
        hspace=HSPACE,
        wspace=WSPACE,
    )

    return fig
