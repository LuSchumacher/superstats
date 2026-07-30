"""Prior push-forward plotting helpers."""

import warnings
from collections.abc import Callable, Mapping
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
    LABEL_PAD,
)

plt.rcParams["axes.axisbelow"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Palatino", "Palatino Linotype", "DejaVu Serif"]

BAND_LABELS = {
    "std": "±1 SD",
    "95ci": "95% CI",
    "mad": "±1.48 MAD",
    "95hdi": "95% HDI",
}


def _select_data_variable(data: Mapping[str, np.ndarray], data_dim: int | str) -> np.ndarray:
    """Resolve named data to a single (batch, steps) variable."""
    if not isinstance(data, Mapping):
        raise TypeError(f"data must be a mapping of named arrays, got {type(data)}.")

    keys = list(data)
    if isinstance(data_dim, int):
        try:
            key = keys[data_dim]
        except IndexError as exc:
            raise ValueError(f"data_dim index {data_dim} is out of range for data keys {keys!r}.") from exc
    else:
        key = data_dim
        if key not in data:
            raise KeyError(f"data key {key!r} not found. Available keys: {keys!r}.")

    x = np.asarray(data[key])
    if x.ndim != 2:
        raise ValueError(f"Data variable {key!r} must have shape (batch_size, steps), got {x.shape}.")
    return x


def plot_push_forward(
    data: Mapping[str, np.ndarray],
    data_dim: int | str = 0,
    kind: Literal["trajectory", "dist"] = "dist",
    aggregation: Callable | None = None,
    uncertainty_fun: Literal["std", "95ci", "mad", "95hdi"] | Callable | None = "95ci",
    marginal: bool = True,
    spaghetti: bool = False,
    alpha: float = 0.5,
    num_cols: int = 3,
    color: str = BASE_COLOR,
    title_fontsize: int = 22,
    label_fontsize: int = 18,
    tick_fontsize: int = 16,
    figsize: tuple[float, float] | None = None,
    max_discrete_values: int = 30,
):
    """Plot prior push-forward for a single data dimension.

    Parameters
    ----------
    data                : mapping of np.ndarray
        Simulation data from the generative model, mapping observation
        names to arrays of shape (batch_size, steps).
    data_dim            : int or str, optional, default: 0
        Which observation variable to plot. Strings select by key and
        integers index the mapping's key order.
    kind                : {"dist", "trajectory"}, optional, default: "dist"
        Plot type: distribution of summary statistics or time-series
        trajectories.
    aggregation         : callable or None, optional, default: None
        Aggregation function over the dataset dimension, called as
        `aggregation(x, axis=...)` (e.g. np.mean, np.median).
        If None, individual datasets are shown in separate panels.
        If specified, all datasets are aggregated into a single panel.
    uncertainty_fun     : {"std", "95ci", "mad", "95hdi"} or callable or None, optional, default: "95ci"
        Uncertainty function. Only used when `aggregation` is not None
        and `kind` is "trajectory". Ignored (with a warning) otherwise.
    marginal            : bool, optional, default: True
        Whether to draw marginal distributions beside trajectory plots.
    spaghetti           : bool, optional, default: False
        Whether to draw individual trajectories behind the aggregate line.
    num_cols            : int, optional, default: 3
        Number of columns when rendering individual panels.
    alpha               : float in [0, 1], optional, default: 0.5
        Alpha value for individual dataset traces.
    color               : str, optional, default: "#822621"
        Base color for plotted lines and fills.
    title_fontsize      : int, optional, default: 22
        The font size of the panel titles.
    label_fontsize      : int, optional, default: 18
        The font size of the axis label texts.
    tick_fontsize       : int, optional, default: 16
        The font size of the axis tick labels.
    figsize            : tuple of two floats or None, optional, default: None
        Explicit figure size in inches. If None, the default layout size
        is used.
    max_discrete_values : int, optional, default: 30
        Maximum number of discrete categories to treat the data as discrete.

    Returns
    -------
    fig : plt.Figure - the figure instance for optional saving

    Raises
    ------
    ValueError
        If `kind` is not "dist" or "trajectory", or if `data` has an
        unsupported shape.
    """
    if kind not in {"dist", "trajectory"}:
        raise ValueError("kind must be 'dist' or 'trajectory'.")

    x = _select_data_variable(data, data_dim)
    show_aggregate = aggregation is not None
    show_uncertainty = uncertainty_fun is not None

    if show_uncertainty and not show_aggregate:
        warnings.warn(
            "uncertainty_fun requires aggregation to be specified; ignoring uncertainty_fun.",
            stacklevel=2,
        )
        uncertainty_fun = None
        show_uncertainty = False

    if show_uncertainty and kind == "dist":
        warnings.warn(
            "uncertainty_fun is not supported for kind='dist'; ignoring uncertainty_fun.",
            stacklevel=2,
        )
        uncertainty_fun = None
        show_uncertainty = False

    batch_size, steps = x.shape
    flat = x.reshape(-1)
    flat = flat[np.isfinite(flat)]
    categories = np.unique(flat)
    discrete = (
        flat.size > 0
        and np.all(np.isclose(categories, np.round(categories)))
        and categories.size <= max_discrete_values
    )

    layout_rect = None

    if show_aggregate:
        if kind == "trajectory":
            t = np.arange(steps)

            default_figsize = (BASE_COL_WIDTH * 1.75, BASE_ROW_HEIGHT * 1.5)
            fig, base_ax = plt.subplots(figsize=figsize if figsize is not None else default_figsize)
            if marginal:
                sub = base_ax.get_subplotspec().subgridspec(1, 2, width_ratios=[4.2, 0.8], wspace=0.0)
                ax = fig.add_subplot(sub[0])
                ax_marg = fig.add_subplot(sub[1])
                base_ax.axis("off")
            else:
                ax = base_ax
                ax_marg = None

            center = aggregation(x, axis=0)

            if show_uncertainty:
                if callable(uncertainty_fun):
                    result = uncertainty_fun(x)
                    if len(result) == 3:
                        center, lower, upper = result
                    elif len(result) == 2:
                        lower, upper = result
                    else:
                        raise ValueError("Custom uncertainty_fun must return (lower, upper) or (center, lower, upper).")
                    center = np.asarray(center)
                    lower = np.asarray(lower)
                    upper = np.asarray(upper)
                elif uncertainty_fun == "95ci":
                    lower = np.percentile(x, 2.5, axis=0)
                    upper = np.percentile(x, 97.5, axis=0)
                elif uncertainty_fun == "std":
                    sd = x.std(axis=0)
                    lower = center - sd
                    upper = center + sd
                elif uncertainty_fun == "mad":
                    med = np.median(x, axis=0)
                    mad = np.median(np.abs(x - med), axis=0)
                    scaled_mad = 1.4826 * mad
                    lower = center - scaled_mad
                    upper = center + scaled_mad
                elif uncertainty_fun == "95hdi":
                    lower, upper = np.empty(steps), np.empty(steps)
                    for i in range(steps):
                        vals = np.sort(x[:, i])
                        n = len(vals)
                        window = int(np.floor(0.95 * n))
                        widths = vals[window:] - vals[: n - window]
                        idx = np.argmin(widths)
                        lower[i], upper[i] = vals[idx], vals[idx + window]
                else:
                    raise ValueError("uncertainty_fun must be 'std', '95ci', 'mad', '95hdi', or callable.")

                ax.fill_between(
                    t,
                    lower,
                    upper,
                    color=color,
                    alpha=0.4,
                    edgecolor="none",
                    zorder=1,
                )

            if spaghetti:
                for i in range(batch_size):
                    ax.plot(
                        t,
                        x[i],
                        color=color,
                        alpha=alpha,
                        linewidth=1.0,
                        zorder=2,
                    )

            ax.plot(
                t,
                center,
                color=color,
                linewidth=2.5,
                zorder=3,
            )
            ax.set_xlabel("Step", fontsize=label_fontsize, labelpad=LABEL_PAD)
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=tick_fontsize)
            if discrete:
                ax.set_yticks(categories)

            if ax_marg is not None:
                values = x.reshape(-1)
                if discrete:
                    counts = np.array([np.sum(values == category) for category in categories])
                    density = counts / counts.sum()
                    ax_marg.barh(categories, density, color=color, alpha=1)
                    ax_marg.set_yticks(categories)
                else:
                    sns.kdeplot(y=values, ax=ax_marg, color=color, fill=True, alpha=1)
                ax_marg.set_ylim(ax.get_ylim())
                ax_marg.set_axis_off()
            aggregate_label = getattr(aggregation, "__name__", "aggregate").capitalize()
            handles = [
                mlines.Line2D([], [], color=color, linewidth=2.5, label=aggregate_label),
            ]
            if show_uncertainty:
                band_label = BAND_LABELS[uncertainty_fun] if isinstance(uncertainty_fun, str) else "Uncertainty"
                handles.append(
                    mpatches.Patch(
                        facecolor=color,
                        alpha=0.4,
                        edgecolor="none",
                        label=band_label,
                    )
                )
            if spaghetti:
                handles.append(
                    mlines.Line2D(
                        [],
                        [],
                        color=color,
                        linewidth=1.5,
                        alpha=1,
                        label="Individual",
                    )
                )
            fig.legend(
                handles=handles,
                loc="lower center",
                ncol=len(handles),
                fontsize=label_fontsize,
                framealpha=0.0,
                bbox_to_anchor=(0.5, -0.02),
            )
            layout_rect = [0, 0.08, 1, 1]

        elif kind == "dist":
            default_figsize = (BASE_COL_WIDTH * 1.75, BASE_ROW_HEIGHT * 1.5)
            fig, ax = plt.subplots(figsize=figsize if figsize is not None else default_figsize)

            if discrete:
                counts = np.array([[np.mean(row.reshape(-1) == category) for category in categories] for row in x])
                heights = aggregation(counts, axis=0)

                ax.bar(categories, heights, color=color, alpha=1)
                ax.set_xticks(categories)
            else:
                stats = aggregation(x, axis=-1)

                sns.histplot(
                    stats,
                    bins=30,
                    stat="density",
                    kde=True,
                    line_kws={"linewidth": 2.0},
                    ax=ax,
                    color=color,
                    alpha=1,
                )

            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=tick_fontsize)

        else:
            raise ValueError("kind must be 'dist' or 'trajectory'.")

    elif kind == "trajectory":
        num_rows = int(np.ceil(batch_size / num_cols))
        default_figsize = (BASE_COL_WIDTH * num_cols, BASE_ROW_HEIGHT * num_rows + 0.5)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize if figsize is not None else default_figsize)
        axes = np.atleast_1d(axes).ravel()
        t = np.arange(steps)

        for i in range(batch_size):
            base_ax = axes[i]
            if marginal:
                sub = base_ax.get_subplotspec().subgridspec(1, 2, width_ratios=[4.2, 0.8], wspace=0.0)
                ax = fig.add_subplot(sub[0])
                ax_marg = fig.add_subplot(sub[1])
                base_ax.axis("off")
            else:
                ax = base_ax
                ax_marg = None

            show_xlabel = i // num_cols == num_rows - 1
            show_ylabel = i % num_cols == 0

            ax.plot(t, x[i], color=color, alpha=1.0, linewidth=1.5)
            ax.set_title(f"Dataset {i}", fontsize=title_fontsize)
            ax.set_xlabel("Step" if show_xlabel else "", fontsize=label_fontsize)
            ax.grid(alpha=0.3)
            ax.tick_params(
                labelsize=tick_fontsize,
                labelbottom=show_xlabel,
                labelleft=show_ylabel,
            )
            if discrete:
                ax.set_yticks(categories)

            if ax_marg is not None:
                if discrete:
                    counts = np.array([np.sum(x[i] == category) for category in categories])
                    density = counts / counts.sum()
                    ax_marg.barh(categories, density, color=color, alpha=1)
                    ax_marg.set_yticks(categories)
                else:
                    sns.kdeplot(y=x[i], ax=ax_marg, color=color, fill=True, alpha=1)
                ax_marg.set_ylim(ax.get_ylim())
                ax_marg.set_axis_off()

        for j in range(batch_size, len(axes)):
            axes[j].axis("off")

    else:
        num_rows = int(np.ceil(batch_size / num_cols))
        default_figsize = (BASE_COL_WIDTH * num_cols, BASE_ROW_HEIGHT * num_rows + 0.5)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize if figsize is not None else default_figsize)
        axes = np.atleast_1d(axes).ravel()

        for i in range(batch_size):
            ax = axes[i]
            if discrete:
                counts = np.array([np.sum(x[i] == category) for category in categories])
                density = counts / counts.sum()
                ax.bar(categories, density, color=color, alpha=1)
                ax.set_xticks(categories)
            else:
                sns.histplot(
                    x[i],
                    bins=30,
                    stat="density",
                    kde=True,
                    line_kws={"linewidth": 2.0},
                    ax=ax,
                    color=color,
                    alpha=1,
                )

            show_xlabel = i // num_cols == num_rows - 1
            show_ylabel = i % num_cols == 0

            ax.set_title(f"Dataset {i}", fontsize=title_fontsize)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.grid(alpha=0.3)
            ax.tick_params(
                labelsize=tick_fontsize,
                labelbottom=show_xlabel,
                labelleft=show_ylabel,
            )

        for j in range(batch_size, len(axes)):
            axes[j].axis("off")

    sns.despine()
    if layout_rect is None:
        plt.tight_layout()
    else:
        plt.tight_layout(rect=layout_rect)

    return fig
