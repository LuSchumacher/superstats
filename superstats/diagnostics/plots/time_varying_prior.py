"""Prior sample visualization helpers."""

from collections.abc import Mapping, Sequence
from itertools import groupby
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from superstats.defaults import (
    BASE_COLOR,
    BASE_COL_WIDTH,
    LABEL_FONTSIZE,
    LABEL_PAD,
    TICK_FONTSIZE,
    TITLE_FONTSIZE,
    Y_LABEL_PAD,
)
from superstats.utils.plotting import (
    PRIOR_ALPHA,
    PRIOR_SECTION_ROW_HEIGHT,
    PRIOR_TITLE_PAD,
    add_prior_heading,
    add_prior_trajectory_legend,
    adjust_prior_layout,
    get_default_num_cols,
    get_layout,
    plot_prior_trajectories,
)


def plot_time_varying_prior(
    local_params: Mapping[str, np.ndarray],
    param_bounds: Mapping[str, tuple[float, float]] | None = None,
    num_cols: int | None = None,
    marginal: bool = True,
    dist_type: Literal["hist", "kde", "both"] = "hist",
    num_bins: int | None = None,
    dist_alpha: float | None = None,
    alpha: float = PRIOR_ALPHA,
    color: str = BASE_COLOR,
    title_fontsize: int = TITLE_FONTSIZE,
    label_fontsize: int = LABEL_FONTSIZE,
    tick_fontsize: int = TICK_FONTSIZE,
    figsize: tuple[float, float] | None = None,
    condition_labels: Mapping[str, Sequence[str]] | None = None,
) -> plt.Figure:
    """Plot time-varying parameter trajectories.

    Individual trajectories and their average are shown for every parameter.
    An optional marginal histogram or KDE is displayed beside each trajectory
    panel. When both raw and implied views are shown, scale headings identify
    their panels.

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
    dist_alpha     : float or None, optional, default: None
        Opacity of marginal distributions. If None, uses 1.0 for a single
        distribution and 0.5 for overlaid conditions.
    alpha          : float, optional, default: 0.25
        Opacity of individual trajectories.
    color          : str, optional, default: BASE_COLOR
        Color used for trajectories and marginals.
    title_fontsize : int, optional, default: 22
        Font size for panel titles and the Trajectories heading.
    label_fontsize : int, optional, default: 18
        Font size for axis labels, scale headings, and the legend.
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
    if not n:
        raise ValueError("No time-varying parameters to plot.")
    if num_cols is None:
        num_cols = get_default_num_cols(n)
    num_rows = int(np.ceil(n / num_cols))
    show_scales = any(name.endswith(" [raw]") for name in local_params) and any(
        name.endswith(" [implied]") for name in local_params
    )
    scale_offset = max(title_fontsize, label_fontsize) + PRIOR_TITLE_PAD + 10
    scale_space = label_fontsize + 10 if show_scales else 0
    heading_offset = scale_offset + scale_space
    heading_height = (heading_offset + title_fontsize + 10) / 72

    plot_figsize, legend_bottom, legend_y = get_layout(
        num_rows,
        num_cols,
        figsize,
        col_width=BASE_COL_WIDTH,
        row_height=PRIOR_SECTION_ROW_HEIGHT + (heading_height - 0.25 + (num_rows - 1) * scale_space / 72) / num_rows,
    )

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=plot_figsize,
    )
    axes = np.atleast_1d(axes).ravel()
    gs = axes[0].get_gridspec()
    panel_scales = []

    for i, (name, values) in enumerate(local_params.items()):
        ax = plot_prior_trajectories(
            axes[i],
            values,
            bounds=(param_bounds or {}).get(name),
            labels=(condition_labels or {}).get(name),
            marginal=marginal,
            dist_type=dist_type,
            num_bins=num_bins,
            dist_alpha=dist_alpha,
            alpha=alpha,
            color=color,
        )
        title = name
        if show_scales:
            title = name.removesuffix(" [raw]").removesuffix(" [implied]")
            view = {" [raw]": "Raw", " [implied]": "Implied"}.get(name[len(title) :], "Raw / Implied")
            panel_scales.append(view)
        ax.set_title(title, fontsize=title_fontsize, pad=PRIOR_TITLE_PAD)

        if i // num_cols == num_rows - 1:
            ax.set_xlabel(
                "Step",
                fontsize=label_fontsize,
                labelpad=LABEL_PAD,
            )

        if i % num_cols == 0:
            ax.set_ylabel(
                "Parameter value",
                fontsize=label_fontsize,
                labelpad=Y_LABEL_PAD,
            )

        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=tick_fontsize)

    for j in range(len(local_params), len(axes)):
        axes[j].axis("off")

    add_prior_trajectory_legend(fig, color, alpha, label_fontsize, legend_y)
    adjust_prior_layout(
        fig,
        gs,
        heading_height=heading_height,
        bottom=legend_bottom,
        min_gap=max(title_fontsize + PRIOR_TITLE_PAD, label_fontsize) + tick_fontsize + 13 + scale_space,
    )
    add_prior_heading(fig, gs[0, :], "Trajectories", heading_offset, title_fontsize)
    for row, start in enumerate(range(0, len(panel_scales), num_cols)):
        column = 0
        for view, panels in groupby(panel_scales[start : start + num_cols]):
            end = column + sum(1 for _ in panels)
            add_prior_heading(fig, gs[row, column:end], view, scale_offset, label_fontsize, "normal")
            column = end
    sns.despine(fig=fig)
    return fig
