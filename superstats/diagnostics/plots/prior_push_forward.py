from __future__ import annotations

from collections.abc import Callable

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

plt.rcParams["axes.axisbelow"] = True


def plot_push_forward(
    data: np.ndarray,
    data_dim: int = 0,
    kind: str = "dist",
    aggregate_fun: str | Callable | None = None,
    uncertainty_fun: str | Callable | None = None,
    marginal: bool = True,
    spaghetti: bool = False,
    n_cols: int = 3,
    color: str = "#822621",
    title_fontsize: int = 14,
    label_fontsize: int = 12,
    tick_fontsize: int = 10,
    alpha: float = 0.5,
    max_discrete_values: int = 30,
):
    """
    Plot prior push-forward for a single data dimension.

    Parameters
    ----------
    data : np.ndarray, shape (batch_size, steps, data_dims)
        Simulation data from the generative model.
    data_dim : int
        Which data dimension to plot.
    kind : {"dist", "trajectory"}
        Plot type: distribution of summary statistics or time-series trajectories.
    aggregate_fun : {"mean", "median"} | callable | None
        Aggregation function over the dataset dimension.
        If None, individual datasets are shown in separate panels.
        If specified, all datasets are aggregated into a single panel.
    uncertainty_fun : {"ci95", "std", "mad"} | callable | None
        Uncertainty function. Only used when aggregate_fun is not None.
    spaghetti : bool
        Whether to draw individual trajectories behind the aggregate line.
    marginal : bool
        Whether to draw marginal distributions beside trajectory plots.
    n_cols : int, default 3
        Number of columns when rendering individual panels.
    color : str, default "#822621"
        Base color for plotted lines and fills.
    title_fontsize : int, default 14
    label_fontsize : int, default 12
    tick_fontsize : int, default 10
    alpha : float, default 0.5
        Alpha value for individual dataset traces.
    max_discrete_values : int, default 30
        Maximum number of discrete categories to treat the data as discrete.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the requested push-forward visualization.
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

    col_width, row_height = 4.0, 3.0
    layout_rect = None

    if show_aggregate:
        if kind == "trajectory":
            t = np.arange(steps)

            fig, base_ax = plt.subplots(figsize=(col_width * 2.5, row_height))
            if marginal:
                sub = base_ax.get_subplotspec().subgridspec(
                    1, 2, width_ratios=[4.2, 0.8], wspace=0.0
                )
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
                        raise ValueError(
                            "Custom uncertainty_fun must return "
                            "(lower, upper) or "
                            "(center, lower, upper)."
                        )

                    center = np.asarray(center)
                    lower = np.asarray(lower)
                    upper = np.asarray(upper)

                elif uncertainty_fun == "ci95":
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

                else:
                    raise ValueError(
                        "uncertainty_fun must be " "'ci95', 'std', 'mad', or callable."
                    )

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
                    counts = np.array(
                        [np.sum(values == category) for category in categories]
                    )
                    density = counts / counts.sum()
                    ax_marg.barh(categories, density, color=color, alpha=1)
                    ax_marg.set_yticks(categories)
                else:
                    sns.kdeplot(y=values, ax=ax_marg, color=color, fill=True, alpha=1)
                ax_marg.set_ylim(ax.get_ylim())
                ax_marg.set_axis_off()

            handles = [
                mlines.Line2D([], [], color="black", linewidth=2.5, label="Average"),
            ]
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
            fig, ax = plt.subplots(figsize=(col_width * 2, row_height))

            if discrete:
                counts = np.array(
                    [
                        [
                            np.mean(row.reshape(-1) == category)
                            for category in categories
                        ]
                        for row in x
                    ]
                )
                if callable(aggregate_fun):
                    heights = np.asarray(aggregate_fun(counts)).reshape(-1)
                elif aggregate_fun == "mean":
                    heights = counts.mean(axis=0)
                elif aggregate_fun == "median":
                    heights = np.median(counts, axis=0)
                else:
                    raise ValueError(
                        "aggregate_fun must be 'mean', 'median', or callable."
                    )

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
                    raise ValueError(
                        "aggregate_fun must be 'mean', 'median', or callable."
                    )

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
        n_rows = int(np.ceil(batch_size / n_cols))
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(col_width * n_cols, row_height * n_rows),
        )
        axes = np.atleast_1d(axes).ravel()
        t = np.arange(steps)

        for i in range(batch_size):
            base_ax = axes[i]
            if marginal:
                sub = base_ax.get_subplotspec().subgridspec(
                    1, 2, width_ratios=[4.2, 0.8], wspace=0.0
                )
                ax = fig.add_subplot(sub[0])
                ax_marg = fig.add_subplot(sub[1])
                base_ax.axis("off")
            else:
                ax = base_ax
                ax_marg = None

            show_xlabel = i // n_cols == n_rows - 1
            show_ylabel = i % n_cols == 0

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
                    counts = np.array(
                        [np.sum(x[i] == category) for category in categories]
                    )
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
        n_rows = int(np.ceil(batch_size / n_cols))
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(col_width * n_cols, row_height * n_rows),
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

            show_xlabel = i // n_cols == n_rows - 1
            show_ylabel = i % n_cols == 0

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
