from collections.abc import Callable
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import seaborn as sns

plt.rcParams["axes.axisbelow"] = True

BAND_LABELS = {"std": "±1 SD", "95ci": "95% CI", "mad": "±1.48 MAD", "95hdi": "95% HDI"}


def _smooth_trajectory(arr: np.ndarray, smoothing: str, window: int) -> np.ndarray:
    """Causal (past-only) smoothing along the last axis."""
    out = arr.copy()
    if smoothing == "sma":
        for i in range(arr.shape[-1]):
            start = max(0, i - window + 1)
            out[..., i] = arr[..., start : i + 1].mean(axis=-1)
    elif smoothing == "ema":
        a = 2.0 / (window + 1)
        out[..., 0] = arr[..., 0]
        for i in range(1, arr.shape[-1]):
            out[..., i] = a * arr[..., i] + (1 - a) * out[..., i - 1]
    else:
        raise ValueError(f"smoothing must be None, 'sma', or 'ema', got {smoothing!r}.")
    return out


def _aggregate_center(
        x: np.ndarray,
        aggregate_fun: str | Callable,
        axis: int = 0
) -> np.ndarray:
    """Reduce x along `axis` using aggregate_fun."""
    if callable(aggregate_fun):
        return np.asarray(aggregate_fun(x))
    if aggregate_fun == "mean":
        return x.mean(axis=axis)
    if aggregate_fun == "median":
        return np.median(x, axis=axis)
    raise ValueError("aggregate_fun must be 'mean', 'median', or callable.")


def _aggregate_label(aggregate_fun: str | Callable | None) -> str:
    """Human-readable label for whatever aggregate_fun resolves to."""
    if aggregate_fun is None:
        return "Median"
    if callable(aggregate_fun):
        return getattr(aggregate_fun, "__name__", "custom").replace("_", " ").capitalize()
    return aggregate_fun.capitalize()


def _compute_uncertainty(
    x: np.ndarray,
    uncertainty_fun: str | Callable,
    center: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """x: (N, T) -> (lo, hi), each (T,). Reduces across axis 0."""
    if callable(uncertainty_fun):
        result = uncertainty_fun(x)
        if len(result) != 2:
            raise ValueError("Custom uncertainty_fun must return (lower, upper).")
        return np.asarray(result[0]), np.asarray(result[1])
    if uncertainty_fun == "std":
        sd = x.std(axis=0)
        return center - sd, center + sd
    if uncertainty_fun == "95ci":
        return np.percentile(x, 2.5, axis=0), np.percentile(x, 97.5, axis=0)
    if uncertainty_fun == "mad":
        mad = np.median(np.abs(x - center), axis=0)
        scaled = 1.4826 * mad
        return center - scaled, center + scaled
    if uncertainty_fun == "95hdi":
        steps = x.shape[-1]
        lo, hi = np.empty(steps), np.empty(steps)
        for i in range(steps):
            vals = np.sort(x[:, i])
            n = len(vals)
            window = int(np.floor(0.95 * n))
            widths = vals[window:] - vals[: n - window]
            idx = np.argmin(widths)
            lo[i], hi[i] = vals[idx], vals[idx + window]
        return lo, hi
    raise ValueError(
        f"uncertainty_fun must be None, 'std', '95ci', 'mad', '95hdi', or callable. "
        f"Got {uncertainty_fun!r}."
    )


def _is_discrete(values: np.ndarray, max_discrete_values: int) -> tuple[np.ndarray, bool]:
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
    pred_data: np.ndarray,
    real_data: np.ndarray,
    data_dim: int = 0,
    kind: Literal["trajectory", "dist"] = "trajectory",
    aggregate_fun: Literal["mean", "median"] | Callable | None = None,
    aggregate_strategy: Literal["full_uncertainty", "no_epistemic"] = "full_uncertainty",
    uncertainty_fun: Literal["std", "95ci", "mad", "95hdi"] | Callable | None = "95hdi",
    smoothing: Literal["sma", "ema"] | None = None,
    smoothing_window: int = 5,
    marginal: bool = True,
    spaghetti: bool = False,
    num_cols: int = 3,
    color: str = "#822621",
    real_color: str = "black",
    alpha: float = 0.4,
    label_fontsize: int = 14,
    tick_fontsize: int = 12,
    max_discrete_values: int = 30,
) -> plt.Figure:
    """
    Plot posterior predictive resimulations against the observed data.

    Parameters
    ----------
    pred_data : np.ndarray, shape (num_datasets, num_resims, num_steps, data_dims)
        Posterior resimulated data.
    real_data : np.ndarray, shape (num_datasets, num_steps, data_dims)
        Observed data.
    data_dim : int
        Which data dimension to plot.
    kind : {"trajectory", "dist"}
        "trajectory": band/center over steps.
        "dist": distribution across steps.
    aggregate_fun : {"mean", "median"} | callable | None
        None: one panel per dataset.
        "mean"/"median"/callable: a single panel aggregated across datasets.
        Also used (instead of a hardcoded median) to collapse resims into a
        per-dataset representative when aggregate_strategy="no_epistemic".
    aggregate_strategy : {"full_uncertainty", "no_epistemic"}
        Only used when aggregate_fun is not None.
        "full_uncertainty": flatten datasets and posterior resims together,
        then summarize. Captures both epistemic and aleatoric uncertainty.
        "no_epistemic": collapse resims to one representative trajectory per
        dataset first (via aggregate_fun), then aggregate across datasets.
        Removes epistemic uncertainty.
    uncertainty_fun : {"std", "95ci", "mad", "95hdi"} | callable | None
        "trajectory" mode only. Function to draw band around the resimulated center line.
    smoothing : {"sma", "ema"} | None
        "trajectory" mode only. Causal (past-only) smoothing applied to the
        real trajectories and, for resimulated data, to the trajectories
        that result *after* aggregate_strategy has pooled resims — i.e.
        pooling happens on raw data, smoothing is applied afterward, and the
        center/uncertainty band are computed on the smoothed result.
    smoothing_window : int
    marginal : bool
        "trajectory" mode only. Attach a marginal KDE panel of the
        resimulated draws to the right of each trajectory axis.
    spaghetti : bool
        "trajectory" mode only. Per-dataset panels: overlay individual resim
        draws behind the band. Aggregated panel: overlay each dataset's own
        representative trajectory (via aggregate_fun) behind the aggregate band.
    num_cols : int
        Number of columns when aggregate_fun is None (per-dataset grid).
    color : str
        Color for bands / centers / histograms.
    real_color : str
        Color for the observed data.
    alpha : float
        Alpha for spaghetti lines.
    label_fontsize : int
    tick_fontsize : int
    max_discrete_values : int
        "dist" mode, per-dataset panels only.
        Maximum number of discrete categories to treat the data as discrete.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if kind not in {"trajectory", "dist"}:
        raise ValueError("kind must be 'trajectory' or 'dist'.")

    pred = np.asarray(pred_data)
    real = np.asarray(real_data)

    if pred.ndim != 4:
        raise ValueError(
            "pred_data must have shape (num_datasets, num_resims, num_steps, data_dims)."
        )
    if real.ndim != 3:
        raise ValueError("real_data must have shape (num_datasets, num_steps, data_dims).")

    D, S, T, _ = pred.shape
    if real.shape[0] != D or real.shape[1] != T:
        raise ValueError("real_data's (num_datasets, num_steps) must match pred_data's.")

    pred_x = pred[..., data_dim]
    real_x = real[..., data_dim]

    if kind == "trajectory" and smoothing is not None:
        real_x = _smooth_trajectory(real_x, smoothing, smoothing_window)

    t = np.arange(T)
    COL_WIDTH, ROW_HEIGHT = 4.0, 3.0
    show_aggregate = aggregate_fun is not None
    agg_label = _aggregate_label(aggregate_fun)

    if kind == "trajectory":

        if show_aggregate:
            fig, base_ax = plt.subplots(figsize=(COL_WIDTH * 2.5, ROW_HEIGHT + 0.5))
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

            # pool resims per aggregate_strategy
            if aggregate_strategy == "full_uncertainty":
                pooled_pred = pred_x.reshape(D * S, T)
            elif aggregate_strategy == "no_epistemic":
                pooled_pred = _aggregate_center(pred_x, aggregate_fun, axis=1)
            else:
                raise ValueError(
                    f"aggregate_strategy must be 'full_uncertainty' or 'no_epistemic', "
                    f"got {aggregate_strategy!r}."
                )

            # smooth the pooled trajectories
            if smoothing is not None:
                pooled_pred = _smooth_trajectory(pooled_pred, smoothing, smoothing_window)

            # aggregate (center) and uncertainty, on the smoothed pool
            center = _aggregate_center(pooled_pred, aggregate_fun, axis=0)
            real_center = _aggregate_center(real_x, aggregate_fun, axis=0)

            if uncertainty_fun is not None:
                lower, upper = _compute_uncertainty(pooled_pred, uncertainty_fun, center)
                ax.fill_between(t, lower, upper, color=color, alpha=0.3, edgecolor="none", zorder=1)

            if spaghetti:
                per_dataset_center = _aggregate_center(pred_x, aggregate_fun, axis=1)
                if smoothing is not None:
                    per_dataset_center = _smooth_trajectory(
                        per_dataset_center, smoothing, smoothing_window
                    )
                for line in per_dataset_center:
                    ax.plot(t, line, color=color, alpha=alpha, linewidth=1.0, zorder=2)

            ax.plot(t, center, color=color, linewidth=2.0, zorder=3)
            ax.plot(t, real_center, color=real_color, linewidth=2.0, zorder=4)

            ax.set_xlabel("Step", fontsize=label_fontsize)
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=tick_fontsize)

            if ax_marg is not None:
                sns.kdeplot(y=pooled_pred.reshape(-1), ax=ax_marg, color=color, fill=True, alpha=1)
                ax_marg.set_ylim(ax.get_ylim())
                ax_marg.set_axis_off()

        else:
            pred_x_panels = (
                _smooth_trajectory(pred_x, smoothing, smoothing_window)
                if smoothing is not None
                else pred_x
            )

            n_rows = int(np.ceil(D / num_cols))
            fig, axes = plt.subplots(
                n_rows,
                num_cols,
                figsize=(COL_WIDTH * num_cols, ROW_HEIGHT * n_rows + 0.5),
            )
            axes = np.atleast_1d(axes).ravel()

            for i in range(D):
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

                pred_traj = pred_x_panels[i]
                real_traj = real_x[i]
                center = np.median(pred_traj, axis=0)

                if uncertainty_fun is not None:
                    lower, upper = _compute_uncertainty(pred_traj, uncertainty_fun, center)
                    ax.fill_between(
                        t, lower, upper, color=color, alpha=0.3, edgecolor="none", zorder=1
                    )

                if spaghetti:
                    for line in pred_traj:
                        ax.plot(t, line, color=color, alpha=alpha, linewidth=1.0, zorder=2)

                ax.plot(t, center, color=color, linewidth=2.0, zorder=3)
                ax.plot(t, real_traj, color=real_color, linewidth=2.0, zorder=4)

                show_xlabel = i // num_cols == n_rows - 1
                ax.set_xlabel("Step" if show_xlabel else "", fontsize=label_fontsize)
                ax.grid(alpha=0.3)
                ax.tick_params(labelsize=tick_fontsize)

                if ax_marg is not None:
                    sns.kdeplot(y=pred_traj.reshape(-1), ax=ax_marg, color=color, fill=True, alpha=1)
                    ax_marg.set_ylim(ax.get_ylim())
                    ax_marg.set_axis_off()

            for j in range(D, len(axes)):
                axes[j].axis("off")

        handles = [
            mlines.Line2D([], [], color=real_color, linewidth=2.0, label="Real data"),
            mlines.Line2D([], [], color=color, linewidth=2.0, label=agg_label),
        ]
        if uncertainty_fun is not None:
            band_label = (
                BAND_LABELS[uncertainty_fun] if isinstance(uncertainty_fun, str) else "Uncertainty"
            )
            handles.append(
                mpatches.Patch(facecolor=color, alpha=0.3, edgecolor="none", label=band_label)
            )
        if spaghetti:
            handles.append(
                mlines.Line2D([], [], color=color, linewidth=1.0, alpha=1, label="Individual")
            )

    else:
        if show_aggregate:
            fig, ax = plt.subplots(figsize=(COL_WIDTH * 2.0, ROW_HEIGHT + 0.5))

            stat_pred = _aggregate_center(pred_x, aggregate_fun, axis=-1)
            stat_real = _aggregate_center(real_x, aggregate_fun, axis=-1)

            if aggregate_strategy == "full_uncertainty":
                pooled_stat = stat_pred.reshape(D * S)
            elif aggregate_strategy == "no_epistemic":
                pooled_stat = _aggregate_center(stat_pred, aggregate_fun, axis=1)
            else:
                raise ValueError(
                    f"aggregate_strategy must be 'full_uncertainty' or 'no_epistemic', "
                    f"got {aggregate_strategy!r}."
                )

            reference = float(_aggregate_center(stat_real, aggregate_fun, axis=0))

            sns.histplot(
                pooled_stat,
                bins=30,
                stat="density",
                kde=True,
                line_kws={"linewidth": 2.0},
                ax=ax,
                color=color,
                alpha=1,
            )
            ax.axvline(reference, color=real_color, linewidth=2.5, linestyle="--", zorder=3)

            ax.set_xlabel(f"{agg_label} statistic", fontsize=label_fontsize)
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=tick_fontsize)

            handles = [
                mpatches.Patch(
                    facecolor=color, alpha=1, edgecolor="none",
                    label=f"Predictive {agg_label.lower()}",
                ),
                mlines.Line2D(
                    [], [], color=real_color, linewidth=2.5, linestyle="--",
                    label=f"Observed {agg_label.lower()}",
                ),
            ]

        else:
            flat = np.concatenate([pred_x.reshape(-1), real_x.reshape(-1)])
            categories, discrete = _is_discrete(flat, max_discrete_values)

            n_rows = int(np.ceil(D / num_cols))
            fig, axes = plt.subplots(
                n_rows,
                num_cols,
                figsize=(COL_WIDTH * num_cols, ROW_HEIGHT * n_rows + 0.5),
            )
            axes = np.atleast_1d(axes).ravel()

            for i in range(D):
                ax = axes[i]
                pred_vals = pred_x[i].reshape(-1)
                real_vals = real_x[i]

                if discrete:
                    width = 0.4
                    pred_freq = np.array([np.mean(pred_vals == c) for c in categories])
                    real_freq = np.array([np.mean(real_vals == c) for c in categories])
                    ax.bar(categories - width / 2, pred_freq, width=width, color=color, alpha=1)
                    ax.bar(
                        categories + width / 2, real_freq, width=width, color=real_color, alpha=1
                    )
                    ax.set_xticks(categories)
                else:
                    sns.histplot(
                        pred_vals,
                        bins=30,
                        stat="density",
                        kde=True,
                        line_kws={"linewidth": 2.0},
                        ax=ax,
                        color=color,
                        alpha=0.5,
                    )
                    sns.kdeplot(real_vals, ax=ax, color=real_color, linewidth=2.0)

                show_xlabel = i // num_cols == n_rows - 1
                ax.set_xlabel("Value" if show_xlabel else "", fontsize=label_fontsize)
                ax.set_ylabel("")
                ax.grid(alpha=0.3)
                ax.tick_params(labelsize=tick_fontsize)

            for j in range(D, len(axes)):
                axes[j].axis("off")

            handles = [
                mpatches.Patch(facecolor=color, alpha=0.7, edgecolor="none", label="Predictive"),
                mlines.Line2D([], [], color=real_color, linewidth=2.0, label="Real data"),
            ]

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        fontsize=label_fontsize,
        framealpha=0.0,
        bbox_to_anchor=(0.5, -0.02),
    )

    sns.despine()
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    return fig