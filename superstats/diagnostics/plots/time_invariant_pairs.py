"""Bivariate time-invariant posterior plots."""

from collections.abc import Mapping, Sequence
from typing import Literal

import bayesflow as bf
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from superstats.defaults import BASE_COLOR, LABEL_FONTSIZE, TICK_FONTSIZE, TITLE_FONTSIZE
from superstats.utils.indexing import format_dataset_label
from superstats.utils.plotting import flatten_time_invariant_parameters, prepare_time_invariant_data

from .time_invariant_marginals import _select_single_dataset


def plot_pairs(
    estimates: Mapping[str, np.ndarray] | np.ndarray,
    targets: Mapping[str, np.ndarray] | np.ndarray | None = None,
    variable_keys: Sequence[str] | None = None,
    variable_names: Sequence[str] | None = None,
    mixture_names: Mapping[str, Sequence[str]] | None = None,
    data_idx: int | None = None,
    dist_type: Literal["hist", "kde", "both"] = "hist",
    height: float = 3.0,
    color: str = BASE_COLOR,
    target_color: str = "black",
    alpha: float = 0.9,
    title_fontsize: int = TITLE_FONTSIZE,
    label_fontsize: int = LABEL_FONTSIZE,
    tick_fontsize: int = TICK_FONTSIZE,
    **kwargs,
) -> plt.Figure:
    """Plot a traditional MCMC pairs plot for one posterior.

    Diagonal panels show univariate marginals, upper panels show draws,
    and lower panels show bivariate densities. Posterior sample and step
    axes are flattened and resampled to the original posterior draw count
    within the selected dataset. An explicit ``data_idx`` is required when
    multiple datasets are present.

    ``dist_type`` is forwarded to the underlying pairs renderer to control
    the diagonal marginal distributions.

    Parameters
    ----------
    estimates : Mapping[str, np.ndarray] or np.ndarray
        Posterior samples. Mapping values must have shape ``(num_datasets,
        num_post_samples, num_steps, num_components)``. Array input must
        have shape ``(num_datasets, num_post_samples, num_steps,
        num_parameters)``.
    targets : Mapping[str, np.ndarray], np.ndarray, or None, optional, default: None
        Time-invariant ground-truth values matching ``estimates``. When
        supplied, targets are shown with cross markers.
    variable_keys : sequence of str or None, optional, default: None
        Variables to plot, and their order, for mapping input. Ignored for
        array input; defaults to every supplied mapping key.
    variable_names : sequence of str or None, optional, default: None
        Display name for each parameter. Defaults to ``variable_keys`` for
        mapping input and ``param_0``, ``param_1``, ... for array input.
    mixture_names : mapping of str to sequence of str or None, optional, default: None
        Component labels for multicomponent mapping variables. Each component
        is displayed as a separate parameter in the pair grid.
    data_idx : int or None, optional, default: None
        Dataset to plot. Required when ``estimates`` contains more than one
        dataset.
    dist_type : {"hist", "kde", "both"}, optional, default: "hist"
        Distribution representation in the diagonal marginal panels.
    height : float, optional, default: 3.0
        Height of each pair-grid panel in inches.
    color : str, optional, default: BASE_COLOR
        Color for posterior draws and densities.
    target_color : str, optional, default: "black"
        Color for target markers.
    alpha : float, optional, default: 0.9
        Opacity of posterior plot elements.
    title_fontsize : int, optional, default: TITLE_FONTSIZE
        Font size of the dataset title when multiple datasets are supplied.
    label_fontsize : int, optional, default: LABEL_FONTSIZE
        Font size of parameter labels and legend text.
    tick_fontsize : int, optional, default: TICK_FONTSIZE
        Font size of axis tick labels.
    **kwargs
        Additional keyword arguments forwarded to BayesFlow's
        ``pairs_posterior`` renderer.

    Returns
    -------
    fig : plt.Figure
        Pair-grid figure with a shared bottom legend.

    Raises
    ------
    ValueError
        If multiple datasets are supplied without ``data_idx`` or if the
        input shapes are invalid.
    TypeError
        If ``data_idx`` is not an integer.

    Notes
    -----
    This function displays one posterior at a time. The posterior sample and
    step axes are never pooled across datasets.
    """
    local_estimates, local_targets, names, local_mixture_names = prepare_time_invariant_data(
        estimates, targets, variable_keys, variable_names, mixture_names
    )
    samples, target_values, parameter_names = flatten_time_invariant_parameters(
        local_estimates, local_targets, names, local_mixture_names
    )
    selected_index = _select_single_dataset(data_idx, samples.shape[0], "plot_pairs")

    grid = bf.diagnostics.plots.pairs_posterior(
        estimates=samples[selected_index],
        targets=target_values[selected_index] if target_values is not None else None,
        variable_names=parameter_names,
        dist_type=dist_type,
        height=height,
        post_color=color,
        target_color=target_color,
        alpha=alpha,
        markersize=60,
        target_markersize=75,
        label_fontsize=label_fontsize,
        tick_fontsize=tick_fontsize,
        **kwargs,
    )
    fig = grid.figure

    # BayesFlow attaches one legend per plot layer at the right edge. Replace
    # those with the shared bottom-legend treatment used by our diagnostics.
    for legend in list(fig.legends):
        legend.remove()

    figure_width, figure_height = fig.get_size_inches()
    legend_space = 1.6
    legend_offset = 0.1
    figure_height += legend_space
    fig.set_size_inches(figure_width, figure_height, forward=True)
    legend_handles = [mpatches.Patch(facecolor=color, edgecolor="none", alpha=alpha, label="Posterior")]
    if target_values is not None:
        legend_handles.append(
            mlines.Line2D([], [], color=target_color, marker="x", linestyle="none", markersize=7, label="Target")
        )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(4, len(legend_handles)),
        fontsize=label_fontsize,
        framealpha=0.0,
        bbox_to_anchor=(0.5, legend_offset / figure_height),
    )
    fig.subplots_adjust(bottom=legend_space / figure_height)

    if samples.shape[0] > 1:
        fig.subplots_adjust(top=0.91)
        fig.suptitle(format_dataset_label(selected_index), fontsize=title_fontsize, y=0.96)
    return fig
