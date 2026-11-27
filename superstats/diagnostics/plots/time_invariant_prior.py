"""Prior sample visualization helpers."""

from collections.abc import Mapping, Sequence
from math import ceil
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
    PRIOR_ROW_HEIGHT,
    PRIOR_SECTION_ROW_HEIGHT,
    PRIOR_TITLE_PAD,
    add_prior_heading,
    adjust_prior_layout,
    get_default_num_cols,
    get_layout,
    plot_prior_distribution,
)


def plot_time_invariant_prior(
    hyper_params: Mapping[str, np.ndarray],
    shared_params: Mapping[str, np.ndarray],
    mixture_names: Mapping[str, Sequence[str]] | None = None,
    dist_type: Literal["hist", "kde", "both"] = "hist",
    num_bins: int | None = None,
    dist_alpha: float | None = None,
    color: str = BASE_COLOR,
    num_cols: int | None = None,
    title_fontsize: int = TITLE_FONTSIZE,
    label_fontsize: int = LABEL_FONTSIZE,
    tick_fontsize: int = TICK_FONTSIZE,
    figsize: tuple[float, float] | None = None,
    condition_labels: Mapping[str, Sequence[str]] | None = None,
) -> plt.Figure:
    """Plot time-invariant hyperparameter and shared-parameter distributions.

    Hyperparameters and shared parameters have separate headings and use the
    same distribution columns. Multiple columns in a sample array are drawn
    as overlaid conditional or mixture-component distributions.

    Parameters
    ----------
    hyper_params   : dict of np.ndarray
        Mapping from hyperparameter names to sample arrays.
    shared_params  : dict of np.ndarray
        Mapping from shared parameter names to sample arrays.
    mixture_names  : dict or None, optional, default: None
        Optional mapping from parameter names to mixture-component labels.
    dist_type      : {"hist", "kde", "both"}, optional, default: "hist"
        Distribution plot type.
    num_bins       : int or None, optional, default: None
        Number of histogram bins. If None, Seaborn selects the bins.
    dist_alpha     : float or None, optional, default: None
        Opacity of parameter distributions. If None, uses 1.0 for a single
        distribution and 0.5 for overlaid mixture components.
    color          : str, optional, default: BASE_COLOR
        Color for non-mixture distributions.
    num_cols       : int or None, optional, default: None
        Number of subplot columns. If ``None``, the larger section determines
        the compact layout:
        1--3 parameters use 1--3 columns, 4 uses 2, 5--6 use 3, 7--8 use
        4, 9 uses 3, and 10 or more use 4.
    title_fontsize : int, optional, default: 22
        Font size for panel titles and section headings.
    label_fontsize : int, optional, default: 18
        Font size for axis labels.
    tick_fontsize  : int, optional, default: 16
        Font size for tick labels and legends.
    figsize        : tuple of two floats or None, optional, default: None
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

    sections = [
        (title, params)
        for title, params in (("Hyperparameters", hyper_params), ("Shared parameters", shared_params))
        if params
    ]
    if not sections:
        raise ValueError("No time-invariant parameters to plot.")
    if num_cols is None:
        num_cols = get_default_num_cols(max(len(hyper_params), len(shared_params)))
    section_rows = [ceil(len(params) / num_cols) for _, params in sections]
    section_gap = len(sections) == 2
    num_plot_rows = sum(section_rows)
    num_rows = num_plot_rows + section_gap
    heading_offset = max(title_fontsize, label_fontsize) + PRIOR_TITLE_PAD + 10
    heading_height = (heading_offset + title_fontsize + 10) / 72
    row_height = PRIOR_SECTION_ROW_HEIGHT if section_gap else PRIOR_ROW_HEIGHT

    plot_figsize, bottom, _ = get_layout(
        num_plot_rows,
        num_cols,
        figsize,
        col_width=BASE_COL_WIDTH,
        row_height=row_height + (heading_height + 0.65 * section_gap) / num_plot_rows,
        legend_space=0.55,
    )

    fig = plt.figure(figsize=plot_figsize)
    row_heights = [1] * num_rows
    gap_row = section_rows[0] if section_gap else None
    if gap_row is not None:
        row_heights[gap_row] = 0
    gs = fig.add_gridspec(num_rows, num_cols, height_ratios=row_heights)

    start_row = 0
    headings = []
    for (title, params), rows in zip(sections, section_rows, strict=True):
        headings.append((gs[start_row, : min(num_cols, len(params))], title))
        items = list(params.items())
        for i in range(rows * num_cols):
            row, col = divmod(i, num_cols)
            ax = fig.add_subplot(gs[start_row + row, col])
            if i >= len(items):
                ax.axis("off")
                continue
            name, values = items[i]
            mixture_owner = name.split("_mixture_weights")[0].strip()
            plot_prior_distribution(
                ax,
                values,
                labels=(condition_labels or {}).get(name),
                component_names=(mixture_names or {}).get(mixture_owner),
                dist_type=dist_type,
                num_bins=num_bins,
                dist_alpha=dist_alpha,
                color=color,
                fontsize=tick_fontsize,
            )
            ax.set_title(name, fontsize=title_fontsize, pad=PRIOR_TITLE_PAD)
            ax.set_xlabel("")
            ax.set_ylabel("")
            if start_row + row == num_rows - 1:
                ax.set_xlabel("Value", fontsize=label_fontsize, labelpad=LABEL_PAD)
            if col == 0:
                ax.set_ylabel("Density", fontsize=label_fontsize, labelpad=Y_LABEL_PAD)
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=tick_fontsize)
        start_row += rows + 1

    adjust_prior_layout(
        fig,
        gs,
        heading_height=heading_height,
        bottom=bottom,
        min_gap=max(title_fontsize + PRIOR_TITLE_PAD, label_fontsize) + tick_fontsize + 13,
        section_gap_row=gap_row,
        section_gap_height=heading_offset + title_fontsize + tick_fontsize + 20,
    )
    for spec, title in headings:
        add_prior_heading(fig, spec, title, heading_offset, title_fontsize)
    sns.despine(fig=fig)
    return fig
