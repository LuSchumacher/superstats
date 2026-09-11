"""Forest plots for collections of time-invariant posteriors."""

from collections.abc import Callable, Mapping, Sequence
from typing import Literal

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from superstats.defaults import (
    BASE_COLOR,
    BASE_COL_WIDTH,
    BASE_ROW_HEIGHT,
    LABEL_FONTSIZE,
    LABEL_PAD,
    TICK_FONTSIZE,
    TITLE_FONTSIZE,
)
from superstats.utils.indexing import format_dataset_label, normalize_data_indices
from superstats.utils.plotting import (
    compute_uncertainty_bands,
    flatten_time_invariant_parameters,
    get_default_num_cols,
    get_layout,
    get_uncertainty_interval_labels,
    prepare_time_invariant_data,
)


def plot_forest(
    estimates: Mapping[str, np.ndarray] | np.ndarray,
    targets: Mapping[str, np.ndarray] | np.ndarray | None = None,
    variable_keys: Sequence[str] | None = None,
    variable_names: Sequence[str] | None = None,
    mixture_names: Mapping[str, Sequence[str]] | None = None,
    interval_probabilities: tuple[float, float] = (0.65, 0.95),
    aggregation: Callable | None = None,
    uncertainty_fun: Literal["std", "ci", "mad", "hdi"] | Callable | None = "hdi",
    num_cols: int | None = None,
    color: str = BASE_COLOR,
    title_fontsize: int = TITLE_FONTSIZE,
    label_fontsize: int = LABEL_FONTSIZE,
    tick_fontsize: int = TICK_FONTSIZE,
    figsize: tuple[float, float] | None = None,
    data_idx: int | Sequence[int] | None = None,
) -> plt.Figure:
    """Compare time-invariant posteriors across datasets with a forest plot.

    With ``aggregation=None`` (the default), each marker is a dataset's
    posterior median. Thick and thin horizontal lines show the inner and
    outer equal-tailed credible intervals, respectively. Posterior sample
    and step axes are flattened and resampled to the original posterior draw
    count within each dataset, never across datasets.

    When ``aggregation`` is a callable, selected datasets are reduced
    independently for every posterior sample. The resulting aggregate draws
    are shown in one forest panel with one row per parameter. In this mode,
    ``uncertainty_fun`` controls the displayed uncertainty bands; named methods
    produce outer and inner intervals, and a callable returns one
    ``(lower, upper)`` interval.

    Parameters
    ----------
    estimates : Mapping[str, np.ndarray] or np.ndarray
        Posterior samples. Mapping values must have shape ``(num_datasets,
        num_post_samples, num_steps, num_components)``. Array input must
        have shape ``(num_datasets, num_post_samples, num_steps,
        num_parameters)``.
    targets : Mapping[str, np.ndarray], np.ndarray, or None, optional, default: None
        Time-invariant ground-truth values matching ``estimates``. When
        supplied, targets are shown with black cross markers.
    variable_keys : sequence of str or None, optional, default: None
        Variables to plot, and their order, for mapping input. Ignored for
        array input; defaults to every supplied mapping key.
    variable_names : sequence of str or None, optional, default: None
        Display name for each variable. Defaults to ``variable_keys`` for
        mapping input and ``param_0``, ``param_1``, ... for array input.
    mixture_names : mapping of str to sequence of str or None, optional, default: None
        Component labels for multicomponent mapping variables. Each component
        receives its own forest panel or aggregate row.
    interval_probabilities : tuple of float, optional, default: (0.65, 0.95)
        Inner and outer equal-tailed credible-interval probabilities used
        when ``aggregation`` is ``None``. Must satisfy ``0 < inner < outer < 1``.
    aggregation : callable or None, optional, default: None
        Reduction applied across selected datasets in aggregate mode, called
        as ``aggregation(values, axis=0)``. ``None`` keeps one row per
        selected dataset. A callable produces one panel with parameters on
        the y-axis.
    uncertainty_fun : {"std", "ci", "mad", "hdi"}, callable, or None, optional, default: "hdi"
        Uncertainty construction in aggregate mode. Named methods draw outer
        and inner bands; a callable returns one ``(lower, upper)`` interval;
        ``None`` omits uncertainty bands. Ignored when ``aggregation`` is
        ``None``.
    num_cols : int or None, optional, default: None
        Number of panel columns in non-aggregate mode. ``None`` selects the
        shared compact layout. Aggregate mode always uses one panel.
    color : str, optional, default: BASE_COLOR
        Color for posterior intervals and center markers.
    title_fontsize : int, optional, default: TITLE_FONTSIZE
        Font size of parameter titles.
    label_fontsize : int, optional, default: LABEL_FONTSIZE
        Font size of axis labels and legend text.
    tick_fontsize : int, optional, default: TICK_FONTSIZE
        Font size of axis tick labels.
    figsize : tuple of float or None, optional, default: None
        Explicit figure size in inches.
    data_idx : int, sequence of int, or None, optional, default: None
        Dataset or datasets to include. ``None`` includes every dataset.

    Returns
    -------
    fig : plt.Figure
        Forest-plot figure.

    Raises
    ------
    ValueError
        If interval probabilities, aggregation output, input shapes, or
        ``num_cols`` are invalid.

    Notes
    -----
    In non-aggregate mode, posterior sample and step axes are flattened and
    resampled independently within each dataset; posterior draws are never
    pooled across datasets.
    """
    inner_probability, outer_probability = interval_probabilities
    if not 0 < inner_probability < outer_probability < 1:
        raise ValueError("interval_probabilities must satisfy 0 < inner < outer < 1.")

    local_estimates, local_targets, names, local_mixture_names = prepare_time_invariant_data(
        estimates, targets, variable_keys, variable_names, mixture_names
    )
    samples, target_values, parameter_names = flatten_time_invariant_parameters(
        local_estimates, local_targets, names, local_mixture_names
    )
    selected_indices = normalize_data_indices(data_idx, samples.shape[0])
    samples = samples[selected_indices]
    if target_values is not None:
        target_values = target_values[selected_indices]

    if aggregation is None:
        quantile_levels = [
            (1 - outer_probability) / 2,
            (1 - inner_probability) / 2,
            0.5,
            1 - (1 - inner_probability) / 2,
            1 - (1 - outer_probability) / 2,
        ]
        outer_low, inner_low, centers, inner_high, outer_high = np.quantile(samples, quantile_levels, axis=1)
        y_labels = [format_dataset_label(index) for index in selected_indices]
        center_label = "Median"
        outer_label = f"{outer_probability:.0%} CI"
        inner_label = f"{inner_probability:.0%} CI"
    else:
        pooled_samples = np.asarray(aggregation(samples, axis=0))

        center = np.asarray(aggregation(pooled_samples, axis=0))
        if center.shape != (samples.shape[-1],):
            raise ValueError(
                f"aggregation must return one value per parameter when called with axis=0; got shape {center.shape}."
            )
        centers = center[None, :]
        if uncertainty_fun is None:
            outer_low = outer_high = inner_low = inner_high = None
        else:
            (outer_low_values, outer_high_values), inner_band = compute_uncertainty_bands(
                pooled_samples, uncertainty_fun, center
            )
            outer_low, outer_high = np.asarray(outer_low_values)[None, :], np.asarray(outer_high_values)[None, :]
            if inner_band is None:
                inner_low = inner_high = None
            else:
                inner_low = np.asarray(inner_band[0])[None, :]
                inner_high = np.asarray(inner_band[1])[None, :]
        if target_values is not None:
            target_values = np.asarray(aggregation(target_values, axis=0))[None, :]
        y_labels = parameter_names
        center_label = getattr(aggregation, "__name__", "aggregate").replace("_", " ").capitalize()
        if uncertainty_fun is None:
            outer_label = inner_label = None
        else:
            outer_label, inner_label = get_uncertainty_interval_labels(uncertainty_fun)

    num_panels = 1 if aggregation is not None else len(parameter_names)
    if aggregation is not None:
        num_cols = 1
    elif num_cols is None:
        num_cols = get_default_num_cols(num_panels)
    if num_cols < 1:
        raise ValueError("num_cols must be at least 1.")
    num_rows = int(np.ceil(num_panels / num_cols))
    row_height = max(BASE_ROW_HEIGHT, 0.42 * len(y_labels) + 1.0)
    if aggregation is not None and figsize is None:
        longest_label = max(len(label) for label in parameter_names)
        label_margin = max(1.5, 0.15 * longest_label + 0.3)
        figsize = (BASE_COL_WIDTH + label_margin, row_height + 1.6)
    plot_figsize, legend_bottom, legend_y = get_layout(
        num_rows, num_cols, figsize, col_width=BASE_COL_WIDTH, row_height=row_height
    )
    fig, axes = plt.subplots(num_rows, num_cols, figsize=plot_figsize, squeeze=False, sharey=True)
    axes_flat = axes.ravel()
    y_positions = np.arange(len(y_labels))

    if aggregation is None:
        for parameter, name in enumerate(parameter_names):
            ax = axes_flat[parameter]
            if outer_low is not None:
                ax.hlines(
                    y_positions,
                    outer_low[:, parameter],
                    outer_high[:, parameter],
                    color=color,
                    linewidth=2.0,
                    alpha=0.55,
                    zorder=2,
                )
            if inner_low is not None:
                ax.hlines(
                    y_positions,
                    inner_low[:, parameter],
                    inner_high[:, parameter],
                    color=color,
                    linewidth=6.0,
                    alpha=0.9,
                    zorder=3,
                )
            ax.scatter(centers[:, parameter], y_positions, color=color, s=60, zorder=4)
            if target_values is not None:
                ax.scatter(
                    target_values[:, parameter],
                    y_positions,
                    color="black",
                    marker="x",
                    s=75,
                    linewidth=1.5,
                    zorder=5,
                )

            ax.set_title(name, fontsize=title_fontsize, pad=15)
            ax.set_xlabel(
                "Value" if parameter // num_cols == num_rows - 1 else "",
                fontsize=label_fontsize,
                labelpad=LABEL_PAD,
            )
            ax.set_yticks(y_positions)
            if parameter % num_cols == 0:
                ax.set_yticklabels(y_labels, fontsize=tick_fontsize)
            ax.tick_params(axis="x", labelsize=tick_fontsize)
            ax.grid(axis="x", alpha=0.3)
    else:
        ax = axes_flat[0]
        if outer_low is not None:
            ax.hlines(
                y_positions,
                outer_low[0],
                outer_high[0],
                color=color,
                linewidth=2.0,
                alpha=0.55,
                zorder=2,
            )
        if inner_low is not None:
            ax.hlines(
                y_positions,
                inner_low[0],
                inner_high[0],
                color=color,
                linewidth=6.0,
                alpha=0.9,
                zorder=3,
            )
        ax.scatter(centers[0], y_positions, color=color, s=60, zorder=4)
        if target_values is not None:
            ax.scatter(
                target_values[0],
                y_positions,
                color="black",
                marker="x",
                s=75,
                linewidth=1.5,
                zorder=5,
            )

        ax.set_xlabel("Value", fontsize=label_fontsize, labelpad=LABEL_PAD)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels, fontsize=tick_fontsize)
        ax.tick_params(axis="x", labelsize=tick_fontsize)
        ax.grid(axis="x", alpha=0.3)

    for panel in range(num_panels, len(axes_flat)):
        axes_flat[panel].axis("off")
    axes_flat[0].invert_yaxis()
    legend_handles = []
    if outer_label is not None:
        legend_handles.append(mlines.Line2D([], [], color=color, linewidth=2.0, alpha=0.55, label=outer_label))
    if inner_low is not None and inner_label is not None:
        legend_handles.append(mlines.Line2D([], [], color=color, linewidth=6.0, alpha=0.9, label=inner_label))
    legend_handles.append(
        mlines.Line2D([], [], color=color, marker="o", linestyle="none", markersize=6, label=center_label)
    )
    if target_values is not None:
        legend_handles.append(
            mlines.Line2D([], [], color="black", marker="x", linestyle="none", markersize=7, label="Target")
        )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(4, len(legend_handles)),
        fontsize=label_fontsize,
        framealpha=0.0,
        bbox_to_anchor=(0.5, legend_y),
    )

    sns.despine()
    plt.tight_layout()
    fig.subplots_adjust(bottom=legend_bottom, hspace=fig.subplotpars.hspace + 0.08)
    return fig
