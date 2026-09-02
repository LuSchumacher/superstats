"""Posterior predictive resimulation plots."""

from collections.abc import Callable, Mapping, Sequence
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
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
from superstats.utils.indexing import format_dataset_label, normalize_data_indices
from superstats.utils.plotting import (
    compute_uncertainty_bands,
    get_default_num_cols,
    get_uncertainty_band_label,
    get_layout,
    plot_dist,
    plot_uncertainty_bands,
    resolve_dist_alpha,
    smooth_trajectories,
)


def _select_resimulation_variable(
    prediction: Mapping[str, np.ndarray],
    empirical: Mapping[str, np.ndarray],
    data_dim: int | str,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve named posterior predictive data to one variable."""
    if not isinstance(prediction, Mapping) or not isinstance(empirical, Mapping):
        raise TypeError("prediction and empirical must be mappings of named arrays.")

    keys = list(prediction)
    if isinstance(data_dim, int):
        try:
            key = keys[data_dim]
        except IndexError as exc:
            raise ValueError(f"data_dim index {data_dim} is out of range for data keys {keys!r}.") from exc
    else:
        key = data_dim
        if key not in prediction:
            raise KeyError(f"prediction key {key!r} not found. Available keys: {keys!r}.")
    if key not in empirical:
        raise KeyError(f"empirical key {key!r} not found. Available keys: {list(empirical)!r}.")

    prediction_x = np.asarray(prediction[key])
    empirical_x = np.asarray(empirical[key])
    if prediction_x.ndim != 3:
        raise ValueError(
            f"Predictive variable {key!r} must have shape "
            f"(num_datasets, num_resims, num_steps), got {prediction_x.shape}."
        )
    if empirical_x.ndim != 2:
        raise ValueError(
            f"Empirical variable {key!r} must have shape (num_datasets, num_steps), got {empirical_x.shape}."
        )
    return prediction_x, empirical_x


def _aggregate_center(x: np.ndarray, aggregation: Callable, axis: int = 0) -> np.ndarray:
    """Reduce x along `axis` using `aggregation`.

    Parameters
    ----------
    x           : np.ndarray
        Array to reduce.
    aggregation : callable
        Reduction to apply, called as `aggregation(x, axis=axis)`.
    axis        : int, optional, default: 0
        Axis to reduce over.

    Returns
    -------
    center : np.ndarray - `x` reduced along `axis`
    """
    return np.asarray(aggregation(x, axis=axis))


def _aggregate_label(aggregation: Callable | None) -> str:
    """Human-readable label for whatever `aggregation` resolves to.

    Parameters
    ----------
    aggregation : callable or None
        None resolves to "Median" (the fixed per-dataset default); a
        callable resolves to its `__name__`.

    Returns
    -------
    label : str - "Median", or a capitalized version of the callable's
        name
    """
    if aggregation is None:
        return "Median"
    return getattr(aggregation, "__name__", "aggregate").replace("_", " ").capitalize()


def _is_discrete(values: np.ndarray, max_discrete_values: int) -> tuple[np.ndarray, bool]:
    """Decide whether pooled values should be treated as a discrete variable.

    Parameters
    ----------
    values              : np.ndarray
        Values to inspect (any shape; flattened internally).
    max_discrete_values : int
        Maximum number of unique integer-like categories to still call
        the variable discrete.

    Returns
    -------
    result : tuple - `(categories, discrete)`, where `categories` is
        the sorted array of unique finite values and `discrete` is
        True if all values are integer-like and there are at most
        `max_discrete_values` unique categories
    """
    flat = values.reshape(-1)
    flat = flat[np.isfinite(flat)]
    categories = np.unique(flat)
    discrete = (
        flat.size > 0
        and np.all(np.isclose(categories, np.round(categories)))
        and categories.size <= max_discrete_values
    )
    return categories, discrete


def plot_posterior_resimulation(
    prediction: Mapping[str, np.ndarray],
    empirical: Mapping[str, np.ndarray],
    data_dim: int | str = 0,
    kind: Literal["time_series", "dist"] = "time_series",
    aggregation: Callable | None = None,
    aggregate_strategy: Literal["full_uncertainty", "no_epistemic"] = "full_uncertainty",
    uncertainty_fun: Literal["std", "ci", "mad", "hdi"] | Callable | None = "hdi",
    smoothing: Literal["sma", "ema"] | None = None,
    smoothing_window: int = 5,
    marginal: bool = True,
    dist_alpha: float | None = None,
    dist_type: Literal["hist", "kde", "both"] = "hist",
    num_bins: int | None = None,
    spaghetti: bool = False,
    num_cols: int | None = None,
    color: str = BASE_COLOR,
    real_color: str = "black",
    alpha: float = 0.4,
    title_fontsize: int = TITLE_FONTSIZE,
    label_fontsize: int = LABEL_FONTSIZE,
    tick_fontsize: int = TICK_FONTSIZE,
    figsize: tuple[float, float] | None = None,
    max_discrete_values: int = 30,
    data_idx: int | Sequence[int] | None = None,
) -> plt.Figure:
    """Plot posterior predictive resimulations against empirical data.

    Parameters
    ----------
    prediction          : mapping of np.ndarray
        Posterior resimulated data, mapping observation names to arrays
        of shape (num_datasets, num_resims, num_steps).
    empirical             : mapping of np.ndarray
        Empirical data, mapping observation names to arrays of shape
        (num_datasets, num_steps).
    data_dim            : int or str, optional, default: 0
        Which observation variable to plot. Strings select by key and
        integers index the predictive mapping's key order.
    kind                : {"time_series", "dist"}, optional, default: "time_series"
        "time_series": band/center over steps.
        "dist": distribution across steps.
    aggregation         : callable or None, optional, default: None
        None: one panel per dataset.
        callable: a single panel aggregated across datasets. Called as
        `aggregation(x, axis=...)` (e.g. np.mean, np.median). Also
        used (instead of a hardcoded median) to collapse resims into
        a per-dataset representative when `aggregate_strategy="no_epistemic"`.
    aggregate_strategy  : {"full_uncertainty", "no_epistemic"}, optional, default: "full_uncertainty"
        Only used when `aggregation` is not None.
        "full_uncertainty": flatten datasets and posterior resims
        together, then summarize. Captures both epistemic and
        aleatoric uncertainty.
        "no_epistemic": collapse resims to one representative
        trajectory per dataset first (via `aggregation`), then
        aggregate across datasets. Removes epistemic uncertainty.
    uncertainty_fun     : {"std", "ci", "mad", "hdi"} or callable or None, optional, default: "hdi"
        "time_series" mode only. Named methods draw nested outer/inner
        ribbons: ±1/±0.5 SD, 95%/65% CI, ±1.48/±0.74 MAD, or
        95%/65% HDI. A callable draws the single interval it returns.
    smoothing           : {"sma", "ema"} or None, optional, default: None
        "time_series" mode only. Causal (past-only) smoothing applied to
        the real trajectories and, for resimulated data, to the
        trajectories that result *after* `aggregate_strategy` has
        pooled resims - i.e. pooling happens on raw data, smoothing is
        applied afterward, and the center/uncertainty band are computed
        on the smoothed result.
    smoothing_window    : int, optional, default: 5
        Window size for `sma`, or span parameter for `ema`.
    marginal            : bool, optional, default: True
        "time_series" mode only. Attach a marginal distribution panel to
        the right of each trajectory axis.
    dist_alpha          : float or None, optional, default: None
        Opacity of predictive and empirical distributions, including
        trajectory marginals. If None, uses 1.0 for a single distribution
        and 0.5 when two distributions are overlaid.
    dist_type           : {"hist", "kde", "both"}, optional, default: "hist"
        Distribution type used for marginals and distribution plots.
    num_bins            : int or None, optional, default: None
        Number of histogram bins. If None, Seaborn selects the bins.
    spaghetti           : bool, optional, default: False
        "time_series" mode only. Per-dataset panels: overlay individual
        resim draws behind the band. Aggregated panel: overlay each
        dataset's own representative trajectory (via `aggregation`)
        behind the aggregate band.
    num_cols            : int or None, optional, default: None
        Exact number of grid columns. If None, uses the shared compact
        dynamic layout based on the selected datasets.
    color               : str, optional, default: BASE_COLOR
        Color for bands / centers / histograms.
    real_color          : str, optional, default: "black"
        Color for the empirical data.
    alpha               : float in [0, 1], optional, default: 0.4
        Alpha for spaghetti lines.
    title_fontsize      : int, optional, default: 22
        The font size of per-dataset panel titles.
    label_fontsize      : int, optional, default: 18
        The font size of the axis label texts.
    tick_fontsize       : int, optional, default: 16
        The font size of the axis tick labels.
    figsize             : tuple of two floats or None, optional, default: None
        Explicit figure size in inches. If None, the default layout size
        is used.
    max_discrete_values : int, optional, default: 30
        "dist" mode, per-dataset panels only. Maximum number of
        discrete categories to treat the data as discrete.
    data_idx            : int, sequence of int, or None, optional, default: None
        Dataset indices to plot. None selects all datasets. A single
        integer still produces a one-dataset panel, and a sequence
        preserves the requested order.

    Returns
    -------
    fig : plt.Figure - the figure instance for optional saving

    Raises
    ------
    ValueError
        If `kind` is not "time_series" or "dist", if `prediction` or
        `empirical` don't have the expected shape, if their
        (num_datasets, num_steps) don't match, or if `aggregate_strategy`
        is not "full_uncertainty" or "no_epistemic".

    Notes
    -----
    ``aggregate_strategy="no_epistemic"`` is a hierarchical collapse of
    the resimulation axis before aggregating datasets. It removes all
    variation along that axis. This isolates epistemic uncertainty only
    when the resimulation axis contains epistemic variation exclusively;
    ordinary posterior-predictive draws may also contain observation noise.
    """
    if kind not in {"time_series", "dist"}:
        raise ValueError("kind must be 'time_series' or 'dist'.")
    if dist_type not in {"hist", "kde", "both"}:
        raise ValueError("dist_type must be one of 'hist', 'kde', or 'both'.")
    if aggregation is not None and aggregate_strategy not in {"full_uncertainty", "no_epistemic"}:
        raise ValueError(
            f"aggregate_strategy must be 'full_uncertainty' or 'no_epistemic', got {aggregate_strategy!r}."
        )
    if num_cols is not None and num_cols < 1:
        raise ValueError("num_cols must be at least 1.")
    prediction_x, empirical_x = _select_resimulation_variable(
        prediction,
        empirical,
        data_dim,
    )

    D, S, T = prediction_x.shape
    if empirical_x.shape[0] != D or empirical_x.shape[1] != T:
        raise ValueError("empirical's (num_datasets, num_steps) must match prediction's.")
    selected_indices = normalize_data_indices(data_idx, D)
    prediction_x = prediction_x[selected_indices]
    empirical_x = empirical_x[selected_indices]
    D = len(selected_indices)
    if num_cols is None:
        num_cols = get_default_num_cols(D)

    if kind == "time_series" and smoothing is not None:
        empirical_x = smooth_trajectories(empirical_x, smoothing, smoothing_window)

    t = np.arange(T)
    show_aggregate = aggregation is not None
    agg_label = _aggregate_label(aggregation)
    dist_alpha = resolve_dist_alpha(
        dist_alpha,
        1 if kind == "dist" and show_aggregate else 2,
    )

    if kind == "time_series":
        has_uncertainty_band = False
        if show_aggregate:
            plot_figsize, legend_bottom, legend_y = get_layout(
                1,
                1,
                figsize,
                col_width=BASE_COL_WIDTH,
                row_height=BASE_ROW_HEIGHT,
            )
            fig, base_ax = plt.subplots(figsize=plot_figsize)
            if marginal:
                sub = base_ax.get_subplotspec().subgridspec(1, 2, width_ratios=[4.2, 0.8], wspace=0.0)
                ax = fig.add_subplot(sub[0])
                ax_marg = fig.add_subplot(sub[1])
                base_ax.axis("off")
            else:
                ax = base_ax
                ax_marg = None

            # pool resims per aggregate_strategy
            if aggregate_strategy == "full_uncertainty":
                pooled_pred = prediction_x.reshape(D * S, T)
            elif aggregate_strategy == "no_epistemic":
                pooled_pred = _aggregate_center(prediction_x, aggregation, axis=1)
            else:
                raise ValueError(
                    f"aggregate_strategy must be 'full_uncertainty' or 'no_epistemic', got {aggregate_strategy!r}."
                )

            # smooth the pooled trajectories
            if smoothing is not None:
                pooled_pred = smooth_trajectories(pooled_pred, smoothing, smoothing_window)

            # aggregate (center) and uncertainty, on the smoothed pool
            center = _aggregate_center(pooled_pred, aggregation, axis=0)
            real_center = _aggregate_center(empirical_x, aggregation, axis=0)

            if uncertainty_fun is not None:
                uncertainty_bands = compute_uncertainty_bands(pooled_pred, uncertainty_fun, center)
                has_uncertainty_band = plot_uncertainty_bands(
                    ax,
                    t,
                    uncertainty_bands[0],
                    uncertainty_bands[1],
                    color,
                    alpha=0.3,
                )

            if spaghetti:
                per_dataset_center = _aggregate_center(prediction_x, aggregation, axis=1)
                if smoothing is not None:
                    per_dataset_center = smooth_trajectories(per_dataset_center, smoothing, smoothing_window)
                for line in per_dataset_center:
                    ax.plot(t, line, color=color, alpha=alpha, linewidth=1.0, zorder=2)

            ax.plot(t, center, color=color, linewidth=2.0, zorder=3)
            ax.plot(t, real_center, color=real_color, linewidth=2.0, linestyle="--", zorder=4)

            ax.set_xlabel(
                "Step",
                fontsize=label_fontsize,
                labelpad=LABEL_PAD,
            )
            ax.set_ylabel(
                "Value",
                fontsize=label_fontsize,
                labelpad=Y_LABEL_PAD,
            )
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=tick_fontsize)

            if ax_marg is not None:
                plot_dist(
                    pooled_pred.reshape(-1),
                    ax=ax_marg,
                    dist_type=dist_type,
                    color=color,
                    orientation="vertical",
                    num_bins=num_bins,
                    alpha=dist_alpha,
                    hide_axis=True,
                )
                empirical_marg_ax = ax_marg.twiny() if dist_type == "hist" else ax_marg
                plot_dist(
                    real_center.reshape(-1),
                    ax=empirical_marg_ax,
                    dist_type=dist_type,
                    color=real_color,
                    orientation="vertical",
                    num_bins=num_bins,
                    alpha=dist_alpha,
                    hide_axis=True,
                )
                ax_marg.set_ylim(ax.get_ylim())
                empirical_marg_ax.set_ylim(ax.get_ylim())
                ax_marg.set_axis_off()
                empirical_marg_ax.set_axis_off()

        else:
            prediction_panels = (
                smooth_trajectories(prediction_x, smoothing, smoothing_window)
                if smoothing is not None
                else prediction_x
            )

            n_rows = int(np.ceil(D / num_cols))
            plot_figsize, legend_bottom, legend_y = get_layout(
                n_rows,
                num_cols,
                figsize,
                col_width=BASE_COL_WIDTH,
                row_height=BASE_ROW_HEIGHT,
            )
            fig, axes = plt.subplots(
                n_rows,
                num_cols,
                figsize=plot_figsize,
            )
            axes = np.atleast_1d(axes).ravel()

            for i in range(D):
                base_ax = axes[i]
                if marginal:
                    sub = base_ax.get_subplotspec().subgridspec(1, 2, width_ratios=[4.2, 0.8], wspace=0.0)
                    ax = fig.add_subplot(sub[0])
                    ax_marg = fig.add_subplot(sub[1])
                    base_ax.axis("off")
                else:
                    ax = base_ax
                    ax_marg = None

                pred_traj = prediction_panels[i]
                real_traj = empirical_x[i]
                center = np.median(pred_traj, axis=0)

                if uncertainty_fun is not None:
                    uncertainty_bands = compute_uncertainty_bands(pred_traj, uncertainty_fun, center)
                    has_uncertainty_band |= plot_uncertainty_bands(
                        ax,
                        t,
                        uncertainty_bands[0],
                        uncertainty_bands[1],
                        color,
                        alpha=0.3,
                    )

                if spaghetti:
                    for line in pred_traj:
                        ax.plot(t, line, color=color, alpha=alpha, linewidth=1.0, zorder=2)

                ax.plot(t, center, color=color, linewidth=2.0, zorder=3)
                ax.plot(t, real_traj, color=real_color, linewidth=2.0, linestyle="--", zorder=4)

                show_xlabel = i // num_cols == n_rows - 1
                show_ylabel = i % num_cols == 0
                ax.set_title(
                    format_dataset_label(selected_indices[i]),
                    fontsize=title_fontsize,
                )
                ax.set_xlabel(
                    "Step" if show_xlabel else "",
                    fontsize=label_fontsize,
                    labelpad=LABEL_PAD,
                )
                ax.set_ylabel(
                    "Value" if show_ylabel else "",
                    fontsize=label_fontsize,
                    labelpad=Y_LABEL_PAD,
                )
                ax.grid(alpha=0.3)
                ax.tick_params(labelsize=tick_fontsize)

                if ax_marg is not None:
                    plot_dist(
                        pred_traj.reshape(-1),
                        ax=ax_marg,
                        dist_type=dist_type,
                        color=color,
                        orientation="vertical",
                        num_bins=num_bins,
                        alpha=dist_alpha,
                        hide_axis=True,
                    )
                    empirical_marg_ax = ax_marg.twiny() if dist_type == "hist" else ax_marg
                    plot_dist(
                        real_traj.reshape(-1),
                        ax=empirical_marg_ax,
                        dist_type=dist_type,
                        color=real_color,
                        orientation="vertical",
                        num_bins=num_bins,
                        alpha=dist_alpha,
                        hide_axis=True,
                    )
                    ax_marg.set_ylim(ax.get_ylim())
                    empirical_marg_ax.set_ylim(ax.get_ylim())
                    ax_marg.set_axis_off()
                    empirical_marg_ax.set_axis_off()

            for j in range(D, len(axes)):
                axes[j].axis("off")

        handles = [
            mlines.Line2D([], [], color=real_color, linewidth=2.0, linestyle="--", label="Empirical"),
            mlines.Line2D([], [], color=color, linewidth=2.0, label=agg_label),
        ]
        if has_uncertainty_band:
            band_label = get_uncertainty_band_label(uncertainty_fun)
            handles.append(mpatches.Patch(facecolor=color, alpha=0.3, edgecolor="none", label=band_label))
        if spaghetti:
            handles.append(mlines.Line2D([], [], color=color, linewidth=1.0, alpha=1, label="Individual"))

    else:
        if show_aggregate:
            plot_figsize, legend_bottom, legend_y = get_layout(
                1,
                1,
                figsize,
                col_width=BASE_COL_WIDTH,
                row_height=BASE_ROW_HEIGHT,
            )
            fig, ax = plt.subplots(figsize=plot_figsize)

            stat_pred = _aggregate_center(prediction_x, aggregation, axis=-1)
            stat_real = _aggregate_center(empirical_x, aggregation, axis=-1)

            if aggregate_strategy == "full_uncertainty":
                pooled_stat = stat_pred.reshape(D * S)
            elif aggregate_strategy == "no_epistemic":
                pooled_stat = _aggregate_center(stat_pred, aggregation, axis=1)
            else:
                raise ValueError(
                    f"aggregate_strategy must be 'full_uncertainty' or 'no_epistemic', got {aggregate_strategy!r}."
                )

            reference = float(_aggregate_center(stat_real, aggregation, axis=0))

            categories, discrete = _is_discrete(
                pooled_stat,
                max_discrete_values,
            )
            if discrete:
                counts = np.array([np.sum(pooled_stat == category) for category in categories])
                heights = counts / counts.sum()
                ax.bar(
                    categories,
                    heights,
                    color=color,
                    alpha=dist_alpha,
                )
                ax.set_xticks(categories)
            else:
                plot_dist(
                    pooled_stat,
                    ax=ax,
                    dist_type=dist_type,
                    color=color,
                    num_bins=num_bins,
                    alpha=dist_alpha,
                )
            ax.axvline(reference, color=real_color, linewidth=2.5, linestyle="--", zorder=3)

            ax.set_xlabel(
                f"{agg_label} statistic",
                fontsize=label_fontsize,
                labelpad=LABEL_PAD,
            )
            ax.set_ylabel(
                "Density",
                fontsize=label_fontsize,
                labelpad=Y_LABEL_PAD,
            )
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=tick_fontsize)

            handles = [
                mpatches.Patch(
                    facecolor=color,
                    alpha=dist_alpha,
                    edgecolor="none",
                    label=f"Predictive {agg_label.lower()}",
                ),
                mlines.Line2D(
                    [],
                    [],
                    color=real_color,
                    linewidth=2.5,
                    linestyle="--",
                    label=f"Empirical {agg_label.lower()}",
                ),
            ]

        else:
            flat = np.concatenate([prediction_x.reshape(-1), empirical_x.reshape(-1)])
            categories, discrete = _is_discrete(flat, max_discrete_values)

            n_rows = int(np.ceil(D / num_cols))
            plot_figsize, legend_bottom, legend_y = get_layout(
                n_rows,
                num_cols,
                figsize,
                col_width=BASE_COL_WIDTH,
                row_height=BASE_ROW_HEIGHT,
            )
            fig, axes = plt.subplots(
                n_rows,
                num_cols,
                figsize=plot_figsize,
                sharex=False,
            )
            axes = np.atleast_1d(axes).ravel()

            for i in range(D):
                ax = axes[i]
                pred_vals = prediction_x[i].reshape(-1)
                real_vals = empirical_x[i]

                if discrete:
                    width = 0.4
                    pred_heights = np.array([np.mean(pred_vals == category) for category in categories])
                    real_heights = np.array([np.mean(real_vals == category) for category in categories])
                    ax.bar(
                        categories - width / 2,
                        pred_heights,
                        width=width,
                        color=color,
                        alpha=dist_alpha,
                    )
                    ax.bar(
                        categories + width / 2,
                        real_heights,
                        width=width,
                        color=real_color,
                        alpha=dist_alpha,
                    )
                    ax.set_xticks(categories)
                else:
                    plot_dist(
                        pred_vals,
                        ax=ax,
                        dist_type=dist_type,
                        color=color,
                        num_bins=num_bins,
                        alpha=dist_alpha,
                    )
                    plot_dist(
                        real_vals,
                        ax=ax,
                        dist_type=dist_type,
                        color=real_color,
                        num_bins=num_bins,
                        alpha=dist_alpha,
                    )

                show_xlabel = i // num_cols == n_rows - 1
                show_ylabel = i % num_cols == 0
                ax.set_title(
                    format_dataset_label(selected_indices[i]),
                    fontsize=title_fontsize,
                )
                ax.set_xlabel(
                    "Value" if show_xlabel else "",
                    fontsize=label_fontsize,
                    labelpad=LABEL_PAD,
                )
                ax.set_ylabel(
                    "Density" if show_ylabel else "",
                    fontsize=label_fontsize,
                    labelpad=Y_LABEL_PAD,
                )
                ax.grid(alpha=0.3)
                ax.tick_params(
                    labelsize=tick_fontsize,
                    labelbottom=True,
                )

            for j in range(D, len(axes)):
                axes[j].axis("off")

            handles = [
                mpatches.Patch(facecolor=color, alpha=dist_alpha, edgecolor="none", label="Predictive"),
                mlines.Line2D([], [], color=real_color, linewidth=2.0, label="Empirical"),
            ]

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
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
