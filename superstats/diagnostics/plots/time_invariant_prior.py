"""Prior sample visualization helpers."""

from collections.abc import Mapping, Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from superstats.defaults import (
    BASE_COLOR,
    BASE_COL_WIDTH,
    BASE_ROW_HEIGHT,
    CATEGORICAL_PALETTE,
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
    resolve_dist_alpha,
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
) -> plt.Figure:
    """Plot time-invariant hyperparameter and shared-parameter distributions.

    Two-dimensional arrays with multiple columns are treated as mixture
    components and plotted separately.

    Parameters
    ----------
    hyper_params   : dict of np.ndarray
        Mapping from hyperparameter names to sample arrays.
    shared_params  : dict of np.ndarray
        Mapping from shared parameter names to sample arrays.
    mixture_names  : dict or None, optional, default: None
        Optional mapping from parameter names to mixture-component labels.
    dist_type      : {"hist", "kde", "both"}, optional, default: "both"
        Distribution plot type.
    num_bins       : int or None, optional, default: None
        Number of histogram bins. If None, Seaborn selects the bins.
    dist_alpha     : float or None, optional, default: None
        Opacity of parameter distributions. If None, uses 1.0 for a single
        distribution and 0.5 for overlaid mixture components.
    color          : str, optional, default: BASE_COLOR
        Color for non-mixture distributions.
    num_cols       : int or None, optional, default: None
        Number of subplot columns. If ``None``, uses a compact default layout:
        1--3 parameters use 1--3 columns, 4 uses 2, 5--6 use 3, 7--8 use
        4, 9 uses 3, and 10 or more use 4.
    title_fontsize : int, optional, default: 22
        Font size for subplot titles.
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
            panel_dist_alpha = resolve_dist_alpha(dist_alpha, arr.shape[1])
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
                "Density",
                fontsize=label_fontsize,
                labelpad=Y_LABEL_PAD,
            )

        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=tick_fontsize)

    for j in range(len(labeled_params), len(axes)):
        axes[j].axis("off")

    sns.despine()
    plt.tight_layout()
    fig.subplots_adjust(
        bottom=legend_bottom,
        hspace=HSPACE,
        wspace=WSPACE,
    )

    return fig
