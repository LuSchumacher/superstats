"""Time-invariant posterior data preparation and marginal plots."""

from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Literal

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
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
from superstats.utils.indexing import format_dataset_label, normalize_data_indices
from superstats.utils.plotting import (
    flatten_time_invariant_parameters,
    prepare_time_invariant_data,
    get_default_num_cols,
    get_layout,
    plot_dist,
    resolve_dist_alpha,
)


def _select_single_dataset(data_idx: int | None, num_datasets: int, plot_name: str) -> int:
    """Resolve exactly one dataset for a posterior-specific plot."""
    if data_idx is None:
        if num_datasets != 1:
            raise ValueError(
                f"{plot_name} displays one posterior at a time. Provide data_idx for {num_datasets} datasets, "
                "or use plot_forest to compare them."
            )
        return 0
    if not isinstance(data_idx, Integral) or isinstance(data_idx, bool):
        raise TypeError("data_idx must be a single integer.")
    return int(normalize_data_indices(int(data_idx), num_datasets)[0])


def plot_marginals(
    estimates: Mapping[str, np.ndarray] | np.ndarray,
    targets: Mapping[str, np.ndarray] | np.ndarray | None = None,
    variable_keys: Sequence[str] | None = None,
    variable_names: Sequence[str] | None = None,
    mixture_names: Mapping[str, Sequence[str]] | None = None,
    dist_type: Literal["hist", "kde", "both"] = "hist",
    num_bins: int | None = None,
    dist_alpha: float | None = None,
    num_cols: int | None = None,
    color: str = BASE_COLOR,
    title_fontsize: int = TITLE_FONTSIZE,
    label_fontsize: int = LABEL_FONTSIZE,
    tick_fontsize: int = TICK_FONTSIZE,
    figsize: tuple[float, float] | None = None,
    data_idx: int | None = None,
) -> plt.Figure:
    """Plot univariate marginals for one time-invariant posterior.

    Posterior sample and step axes are flattened within the selected dataset
    and resampled without replacement to the original posterior draw count.

    Parameters
    ----------
    estimates : Mapping[str, np.ndarray] or np.ndarray
        Posterior samples. Mapping values must have shape ``(num_datasets,
        num_post_samples, num_steps, num_components)``. Array input must
        have shape ``(num_datasets, num_post_samples, num_steps,
        num_parameters)``.
    targets : Mapping[str, np.ndarray], np.ndarray, or None, optional, default: None
        Time-invariant ground-truth values matching ``estimates``. When
        supplied, targets are drawn as black dashed vertical lines.
    variable_keys : sequence of str or None, optional, default: None
        Variables to plot, and their order, for mapping input. Ignored for
        array input; defaults to every supplied mapping key.
    variable_names : sequence of str or None, optional, default: None
        Display name for each variable. Defaults to ``variable_keys`` for
        mapping input and ``param_0``, ``param_1``, ... for array input.
    mixture_names : mapping of str to sequence of str or None, optional, default: None
        Component labels for multicomponent mapping variables. Components are
        overlaid and shown in the figure legend.
    dist_type : {"hist", "kde", "both"}, optional, default: "hist"
        Distribution representation for each marginal.
    num_bins : int or None, optional, default: None
        Number of histogram bins. Seaborn chooses the bins when ``None``.
    dist_alpha : float or None, optional, default: None
        Opacity of the posterior distributions. ``None`` uses ``1.0`` for a
        single component and ``0.5`` for overlaid mixture components.
    num_cols : int or None, optional, default: None
        Number of columns in the panel grid. ``None`` selects the shared
        compact layout.
    color : str, optional, default: BASE_COLOR
        Color for non-mixture distributions.
    title_fontsize : int, optional, default: TITLE_FONTSIZE
        Font size of panel titles.
    label_fontsize : int, optional, default: LABEL_FONTSIZE
        Font size of axis labels and legend text.
    tick_fontsize : int, optional, default: TICK_FONTSIZE
        Font size of axis tick labels.
    figsize : tuple of float or None, optional, default: None
        Explicit figure size in inches.
    data_idx : int or None, optional, default: None
        Dataset to plot. Required when ``estimates`` contains more than one
        dataset.

    Returns
    -------
    fig : plt.Figure
        Figure containing one marginal panel per parameter.

    Raises
    ------
    ValueError
        If multiple datasets are supplied without ``data_idx``, if the input
        shapes are invalid, or if ``num_cols`` is less than one.
    TypeError
        If ``data_idx`` is not an integer.

    Notes
    -----
    This function displays one posterior at a time. Use :func:`plot_forest`
    to compare the same parameters across datasets.
    """
    local_estimates, local_targets, names, local_mixture_names = prepare_time_invariant_data(
        estimates, targets, variable_keys, variable_names, mixture_names
    )
    samples, target_values, _ = flatten_time_invariant_parameters(
        local_estimates, local_targets, names, local_mixture_names
    )
    num_datasets = samples.shape[0]
    selected_index = _select_single_dataset(data_idx, num_datasets, "plot_marginals")

    num_panels = len(names)
    if num_cols is None:
        num_cols = get_default_num_cols(num_panels)
    if num_cols < 1:
        raise ValueError("num_cols must be at least 1.")
    num_rows = int(np.ceil(num_panels / num_cols))
    plot_figsize, legend_bottom, legend_y = get_layout(
        num_rows, num_cols, figsize, col_width=BASE_COL_WIDTH, row_height=BASE_ROW_HEIGHT
    )
    fig, axes = plt.subplots(num_rows, num_cols, figsize=plot_figsize, squeeze=False)
    axes_flat = axes.ravel()

    legend_handles = []
    legend_labels = set()
    component_offset = 0
    for panel, name in enumerate(names):
        ax = axes_flat[panel]
        num_components = local_estimates[name].shape[-1]
        values = samples[selected_index, :, component_offset : component_offset + num_components]
        component_labels = local_mixture_names.get(name, [f"component {i}" for i in range(num_components)])
        is_mixture = num_components > 1
        panel_dist_alpha = resolve_dist_alpha(dist_alpha, num_components)
        for component, component_label in enumerate(component_labels):
            component_color = CATEGORICAL_PALETTE[component % len(CATEGORICAL_PALETTE)] if is_mixture else color
            plot_dist(
                values[..., component].reshape(-1),
                ax=ax,
                dist_type=dist_type,
                color=component_color,
                num_bins=num_bins,
                alpha=panel_dist_alpha,
            )
            if is_mixture and component_label not in legend_labels:
                legend_handles.append(
                    mpatches.Patch(
                        facecolor=component_color,
                        edgecolor="none",
                        alpha=panel_dist_alpha,
                        label=component_label,
                    )
                )
                legend_labels.add(component_label)
            if local_targets is not None:
                ax.axvline(
                    target_values[selected_index, component_offset + component],
                    color="black",
                    linestyle="--",
                    linewidth=1.5,
                    zorder=5,
                )

        component_offset += num_components

        ax.set_title(name, fontsize=title_fontsize, pad=15)
        ax.set_ylabel("Density" if panel % num_cols == 0 else "", fontsize=label_fontsize, labelpad=Y_LABEL_PAD)
        ax.set_xlabel("Value" if panel // num_cols == num_rows - 1 else "", fontsize=label_fontsize, labelpad=LABEL_PAD)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=tick_fontsize)

    for panel in range(num_panels, len(axes_flat)):
        axes_flat[panel].axis("off")
    if local_targets is not None:
        legend_handles.append(mlines.Line2D([], [], color="black", linestyle="--", linewidth=1.5, label="Target"))
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=min(4, len(legend_handles)),
            fontsize=label_fontsize,
            framealpha=0.0,
            bbox_to_anchor=(0.5, legend_y),
        )

    if num_datasets > 1:
        fig.suptitle(format_dataset_label(selected_index), fontsize=title_fontsize)
    sns.despine()
    plt.tight_layout()
    fig.subplots_adjust(bottom=legend_bottom, hspace=HSPACE, wspace=WSPACE)
    return fig
