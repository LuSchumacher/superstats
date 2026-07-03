import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import seaborn as sns
from collections.abc import Callable
from typing import Literal

plt.rcParams["axes.axisbelow"] = True

PALETTE = [
    "#822621",
    "#C1440E",
    "#E8871A",
    "#D4A843",
]


def plot_time_varying_posterior(
    samples: dict,
    local_keys: list[str],
    aggregate_fun: Literal["mean", "median"] | Callable | None = None,
    aggregate_strategy: Literal["full_uncertainty", "no_epistemic"] = "full_uncertainty",
    uncertainty_fun: Literal["std", "95ci", "mad", "95hdi"] | Callable | None = "95ci",
    smoothing: Literal["sma", "ema"] | None = None,
    smoothing_window: int = 5,
    marginal: bool = True,
    n_cols: int = 2,
    color: str = "#822621",
    alpha: float = 0.5,
    title_fontsize: int = 16,
    label_fontsize: int = 14,
    tick_fontsize: int = 12,
) -> plt.Figure:
    """
    Plot time-varying parameter posteriors.

    Parameters
    ----------
    samples : dict
        Each value has shape (num_datasets, num_post_samples, num_steps, 1).
    local_keys : list of str
    aggregate_fun : {"mean", "median"} | callable | None
        None: one panel per (param, dataset).
        "mean"/"median"/callable: one panel per param, aggregated across datasets.
        Callable receives (N, T) trajectories and must return (T,) center.
    aggregate_strategy : {"full_uncertainty", "no_epistemic"}
        Only used when aggregate_fun is not None.
        "full_uncertainty": flatten datasets and posterior samples, then summarize.
        "no_epistemic": median across posterior samples per dataset first, then aggregate.
    uncertainty_fun : {"std", "95ci", "mad", "95hdi"} | callable | None
        Callable receives (N, T) trajectories and must return (lo, hi) each of shape (T,).
    smoothing : {"sma", "ema"} | None
        Applied to each trajectory before computing center, uncertainty, and marginal.
    smoothing_window : int
    marginal : bool
        Attach a marginal KDE panel to the right of each trajectory axis.
        KDE is computed on the same array used for uncertainty.
    """
    D, S, T, _ = next(iter(samples.values())).shape
    P = len(local_keys)
    t = np.arange(T)

    _BAND_LABELS = {"std": "±1 SD", "95ci": "95% CI", "mad": "±1.48 MAD", "95hdi": "95% HDI"}

    # layout
    if aggregate_fun is None:
        fig, axes = plt.subplots(P, D, figsize=(4.5 * D, 3.0 * P), squeeze=False)
    else:
        n_rows = int(np.ceil(P / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 3.0 * n_rows), squeeze=False)

    axes_flat = axes.ravel()

    # per-panel loop
    panel = 0
    for name in local_keys:
        datasets = range(D) if aggregate_fun is None else [None]
        for d in datasets:

            # prepare trajectories (N, T)
            if aggregate_fun is None:
                trajectories = samples[name][d, :, :, 0]
            else:
                param = samples[name][:, :, :, 0]
                if aggregate_strategy == "full_uncertainty":
                    trajectories = param.reshape(D * S, T)
                elif aggregate_strategy == "no_epistemic":
                    trajectories = np.median(param, axis=1)
                else:
                    raise ValueError(f"Unknown aggregate_strategy: {aggregate_strategy!r}")

            # smooth each trajectory before summarizing
            if smoothing == "sma":
                smoothed = trajectories.copy()
                for i in range(T):
                    smoothed[:, i] = trajectories[:, max(0, i - smoothing_window + 1) : i + 1].mean(axis=1)
                trajectories = smoothed
            elif smoothing == "ema":
                smoothed = trajectories.copy()
                a = 2.0 / (smoothing_window + 1)
                for i in range(1, T):
                    smoothed[:, i] = a * trajectories[:, i] + (1 - a) * smoothed[:, i - 1]
                trajectories = smoothed

            # center
            if aggregate_fun is None:
                center = np.median(trajectories, axis=0)
            elif callable(aggregate_fun):
                center = np.asarray(aggregate_fun(trajectories))
            elif aggregate_fun == "mean":
                center = trajectories.mean(axis=0)
            elif aggregate_fun == "median":
                center = np.median(trajectories, axis=0)
            else:
                raise ValueError(f"Unknown aggregate_fun: {aggregate_fun!r}")

            # uncertainty bands
            lo, hi = None, None
            if callable(uncertainty_fun):
                lo, hi = uncertainty_fun(trajectories)
                lo, hi = np.asarray(lo), np.asarray(hi)
            elif uncertainty_fun == "std":
                sd     = trajectories.std(axis=0)
                lo, hi = center - sd, center + sd
            elif uncertainty_fun == "95ci":
                lo, hi = np.percentile(trajectories, 2.5, axis=0), np.percentile(trajectories, 97.5, axis=0)
            elif uncertainty_fun == "mad":
                mad    = np.median(np.abs(trajectories - center), axis=0)
                lo, hi = center - 1.4826 * mad, center + 1.4826 * mad
            elif uncertainty_fun == "95hdi":
                lo, hi = np.empty(T), np.empty(T)
                for i in range(T):
                    vals   = np.sort(trajectories[:, i])
                    n      = len(vals)
                    window = int(np.floor(0.95 * n))
                    widths = vals[window:] - vals[:n - window]
                    idx    = np.argmin(widths)
                    lo[i], hi[i] = vals[idx], vals[idx + window]
            elif uncertainty_fun is not None:
                raise ValueError(f"Unknown uncertainty_fun: {uncertainty_fun!r}")

            # axes setup
            ax_base = axes_flat[panel]
            if marginal:
                sub    = ax_base.get_subplotspec().subgridspec(1, 2, width_ratios=[4.2, 0.8], wspace=0.0)
                ax     = fig.add_subplot(sub[0])
                ax_kde = fig.add_subplot(sub[1])
                ax_base.axis("off")
            else:
                ax     = ax_base
                ax_kde = None

            # plot
            if lo is not None:
                ax.fill_between(t, lo, hi, color=color, alpha=alpha, edgecolor="none")
            ax.plot(t, center, color=color, linewidth=1.5)

            if aggregate_fun is None:
                if panel < D:
                    ax.set_title(f"Dataset {d}", fontsize=title_fontsize)
                if d == 0:
                    ax.set_ylabel(name, fontsize=label_fontsize, rotation=0, labelpad=20)
                if panel >= (P - 1) * D:
                    ax.set_xlabel("Step", fontsize=label_fontsize)
            else:
                ax.set_title(name, fontsize=title_fontsize)
                ax.set_xlabel("Step", fontsize=label_fontsize)

            ax.tick_params(labelsize=tick_fontsize)
            ax.grid(alpha=0.3)

            # marginal KDE
            if marginal:
                sns.kdeplot(y=trajectories.ravel(), ax=ax_kde, color=color, fill=True, alpha=1)
                ax_kde.set_ylim(ax.get_ylim())
                ax_kde.set_axis_off()

            panel += 1

    for j in range(panel, len(axes_flat)):
        axes_flat[j].axis("off")

    # legend
    handles = [mlines.Line2D([], [], color=color, linewidth=1.5, label="Median")]
    if uncertainty_fun is not None:
        band_label = "Uncertainty band" if callable(uncertainty_fun) else _BAND_LABELS[uncertainty_fun]
        handles.append(mpatches.Patch(color=color, alpha=alpha, label=band_label))

    fig.legend(
        handles=handles, loc="lower center", ncol=len(handles),
        fontsize=label_fontsize, framealpha=0.0, bbox_to_anchor=(0.5, -0.02)
    )
    sns.despine()
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    return fig


def plot_time_invariant_posterior(
    samples: dict,
    keys: list[str],
    aggregate: bool = False,
    mixture_names: dict | None = None,
    num_out: int | None = None,
    rng: np.random.Generator | None = None,
    n_cols: int = 2,
    color: str = "#822621",
    title_fontsize: int = 16,
    tick_fontsize: int = 12,
) -> plt.Figure:
    """
    Plot time-invariant parameter posteriors.

    Parameters
    ----------
    samples : dict
        Each value has shape (num_datasets, num_post_samples, num_steps, num_components).
    keys : list of str
        Parameter names to plot (hyper_keys + shared_keys).
    aggregate : bool
        False: rows=params, cols=datasets, param name as row label, dataset index as col title.
        True: one panel per param in a grid.
    mixture_names : dict, optional
        Mapping from parameter name to a list of component names.
        Defaults to "component 0", "component 1", ... when not supplied.
    num_out : int | None
        Number of samples to draw after pooling S and T. Defaults to num_post_samples.
    rng : np.random.Generator | None
    n_cols : int
        Number of columns when aggregate=True.
    color : str
        Base color for non-mixture parameters.
    title_fontsize : int
    tick_fontsize : int
    """
    rng = np.random.default_rng(rng)
    mixture_names = mixture_names or {}
    D = next(iter(samples.values())).shape[0]

    # panels meta
    panels_meta = []
    for name in keys:
        n_components = samples[name].shape[-1]
        if n_components > 1:
            comp_names = mixture_names.get(
                name, [f"component {i}" for i in range(n_components)]
            )
            panels_meta.append((name, list(range(n_components)), comp_names, True))
        else:
            panels_meta.append((name, [0], [name], False))

    P = len(panels_meta)

    # pool across steps
    pooled = {}
    for name in keys:
        arr = samples[name]
        B, S, T, C = arr.shape
        n = num_out if num_out is not None else S
        for c in range(C):
            flat = arr[:, :, :, c].reshape(B, S * T)
            idx  = rng.integers(0, S * T, size=(B, n))
            pooled[(name, c)] = flat[np.arange(B)[:, None], idx]

    # layout
    if not aggregate:
        fig, axes = plt.subplots(P, D, figsize=(4.5 * D, 3.0 * P), squeeze=False)
    else:
        n_rows = int(np.ceil(P / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 3.0 * n_rows), squeeze=False)

    axes_flat = axes.ravel()

    # per-panel loop
    panel = 0
    legend_drawn = False
    for p, (param_name, comp_indices, comp_labels, is_mixture) in enumerate(panels_meta):
        datasets = range(D) if not aggregate else [None]
        for d in datasets:
            ax = axes_flat[panel]

            for ci, (c, label) in enumerate(zip(comp_indices, comp_labels)):
                c_color = PALETTE[ci % len(PALETTE)] if is_mixture else color
                if not aggregate:
                    sns.kdeplot(x=pooled[(param_name, c)][d], ax=ax, color=c_color,
                                fill=True, alpha=1, label=label if is_mixture else None)
                else:
                    sns.kdeplot(x=pooled[(param_name, c)].ravel(), ax=ax, color=c_color,
                                fill=True, alpha=1, label=label if is_mixture else None)

            if not aggregate:
                if p == 0:
                    ax.set_title(f"Dataset {d}", fontsize=title_fontsize)
                if d == 0:
                    ax.set_ylabel(param_name, fontsize=title_fontsize, rotation=0, labelpad=80)
                else:
                    ax.set_ylabel("")
            else:
                ax.set_title(param_name, fontsize=title_fontsize)
                ax.set_ylabel("")

            ax.set_xlabel("")

            if is_mixture and not legend_drawn:
                ax.legend(fontsize=tick_fontsize, framealpha=0.0)
                legend_drawn = True

            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=tick_fontsize)
            panel += 1

    for j in range(panel, len(axes_flat)):
        axes_flat[j].axis("off")

    sns.despine()
    plt.tight_layout()
    return fig