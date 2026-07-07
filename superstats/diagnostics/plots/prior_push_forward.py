from collections.abc import Callable
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import seaborn as sns

plt.rcParams["axes.axisbelow"] = True

BAND_LABELS = {"std": "±1 SD", "95ci": "95% CI", "mad": "±1.48 MAD", "95hdi": "95% HDI"}


def plot_push_forward(
    data: np.ndarray,
    data_dim: int = 0,
    kind: Literal["trajectory", "dist"] = "dist",
    aggregate_fun: Literal["mean", "median"] | Callable | None = None,
    uncertainty_fun: Literal["std", "95ci", "mad", "95hdi"] | Callable | None = "95ci",
    marginal: bool = True,
    spaghetti: bool = False,
    num_cols: int = 3,
    color: str = "#822621",
    title_fontsize: int = 16,
    label_fontsize: int = 14,
    tick_fontsize: int = 12,
    alpha: float = 0.5,
    max_discrete_values: int = 30,
):
    """Plot prior push-forward for a single data dimension.

    Parameters
    ----------
    data                : np.ndarray of shape (batch_size, steps, data_dims)
        Simulation data from the generative model.
    data_dim            : int, optional, default: 0
        Which data dimension to plot.
    kind                : {"dist", "trajectory"}, optional, default: "dist"
        Plot type: distribution of summary statistics or time-series
        trajectories.
    aggregate_fun       : {"mean", "median"} or callable or None, optional, default: None
        Aggregation function over the dataset dimension.
        If None, individual datasets are shown in separate panels.
        If specified, all datasets are aggregated into a single panel.
    uncertainty_fun     : {"std", "95ci", "mad", "95hdi"} or callable or None, optional, default: "95ci"
        Uncertainty function. Only used when `aggregate_fun` is not None.
    marginal            : bool, optional, default: True
        Whether to draw marginal distributions beside trajectory plots.
    spaghetti           : bool, optional, default: False
        Whether to draw individual trajectories behind the aggregate line.
    num_cols            : int, optional, default: 3
        Number of columns when rendering individual panels.
    color               : str, optional, default: "#822621"
        Base color for plotted lines and fills.
    title_fontsize      : int, optional, default: 16
        The font size of the panel titles.
    label_fontsize      : int, optional, default: 14
        The font size of the axis label texts.
    tick_fontsize       : int, optional, default: 12
        The font size of the axis tick labels.
    alpha               : float in [0, 1], optional, default: 0.5
        Alpha value for individual dataset traces.
    max_discrete_values : int, optional, default: 30
        Maximum number of discrete categories to treat the data as discrete.

    Returns
    -------
    fig : plt.Figure - the figure instance for optional saving

    Raises
    ------
    ValueError
        If `kind` is not "dist" or "trajectory", if `uncertainty_fun`
        is given without `aggregate_fun`, or if `aggregate_fun` or
        `uncertainty_fun` (when given as a string) is not one of the
        recognized values.
    """
    if kind not in {"dist", "trajectory"}:
        raise ValueError("kind must be 'dist' or 'trajectory'.")

    x = np.asarray(data)[:, :, data_dim]
    show_aggregate = aggregate_fun is not None
    show_uncertainty = uncertainty_fun is not None

    if show_uncertainty and not show_aggregate:
        raise ValueError("uncertainty_fun requires aggregate_fun to be specified.")
    batch_size, steps = x.shape
    flat = x.reshape(-1)
    flat = flat[np.isfinite(flat)]
    categories = np.unique(flat)
    discrete = (
        flat.size > 0
        and np.all(np.isclose(categories, np.round(categories)))
        and categories.size <= max_discrete_values
    )

    COL_WIDTH, ROW_HEIGHT = 4.0, 3.0
    layout_rect = None

    if show_aggregate:
        if kind == "trajectory":
            t = np.arange(steps)

            fig, base_ax = plt.subplots(figsize=(COL_WIDTH * 2.5, ROW_HEIGHT + 0.5))
            if marginal:
                sub = base_ax.get_subplotspec().subgridspec(1, 2, width_ratios=[4.2, 0.8], wspace=0.0)
                ax = fig.add_subplot(sub[0])
                ax_marg = fig.add_subplot(sub[1])
                base_ax.axis("off")
            else:
                ax = base_ax
                ax_marg = None

            if callable(aggregate_fun):
                center = np.asarray(aggregate_fun(x))
            elif aggregate_fun == "mean":
                center = x.mean(axis=0)
            elif aggregate_fun == "median":
                center = np.median(x, axis=0)
            else:
                raise ValueError("aggregate_fun must be 'mean', 'median', or callable.")

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
                color="black",
                linewidth=2.5,
                zorder=3,
            )
            ax.set_xlabel("Step", fontsize=label_fontsize)
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

            handles = [
                mlines.Line2D([], [], color="black", linewidth=2.5, label="Aggregate"),
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
            fig, ax = plt.subplots(figsize=(COL_WIDTH * 2.0, ROW_HEIGHT + 0.5))

            if discrete:
                counts = np.array([[np.mean(row.reshape(-1) == category) for category in categories] for row in x])
                if callable(aggregate_fun):
                    heights = np.asarray(aggregate_fun(counts)).reshape(-1)
                elif aggregate_fun == "mean":
                    heights = counts.mean(axis=0)
                elif aggregate_fun == "median":
                    heights = np.median(counts, axis=0)
                else:
                    raise ValueError("aggregate_fun must be 'mean', 'median', or callable.")

                ax.bar(categories, heights, color=color, alpha=1)
                ax.set_xticks(categories)
            else:
                if callable(aggregate_fun):
                    stats = np.asarray(aggregate_fun(x)).reshape(-1)
                elif aggregate_fun == "mean":
                    stats = x.mean(axis=-1)
                elif aggregate_fun == "median":
                    stats = np.median(x, axis=-1)
                else:
                    raise ValueError("aggregate_fun must be 'mean', 'median', or callable.")

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
        n_rows = int(np.ceil(batch_size / num_cols))
        fig, axes = plt.subplots(
            n_rows,
            num_cols,
            figsize=(COL_WIDTH * num_cols, ROW_HEIGHT * n_rows + 0.5),
        )
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

            show_xlabel = i // num_cols == n_rows - 1
            show_ylabel = i % num_cols == 0

            ax.plot(t, x[i], color=color, alpha=alpha, linewidth=1.5)
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
        n_rows = int(np.ceil(batch_size / num_cols))
        fig, axes = plt.subplots(
            n_rows,
            num_cols,
            figsize=(COL_WIDTH * num_cols, ROW_HEIGHT * n_rows + 0.5),
        )
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

            show_xlabel = i // num_cols == n_rows - 1
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
