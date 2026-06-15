import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Callable


def _apply_smoothing(arr: np.ndarray, smoothing: str | None, window: int) -> np.ndarray:
    """
    Apply causal (past-only) smoothing to a (..., steps) array along the last axis.

    Parameters
    ----------
    arr : np.ndarray, shape (..., steps)
    smoothing : {None, "sma", "ema"}
    window : int
        For SMA: number of past steps to average (including current).
        For EMA: controls the span — alpha = 2 / (window + 1).
    """
    if smoothing is None:
        return arr

    steps = arr.shape[-1]
    out   = arr.copy()

    if smoothing == "sma":
        for t in range(steps):
            start      = max(0, t - window + 1)
            out[..., t] = arr[..., start : t + 1].mean(axis=-1)

    elif smoothing == "ema":
        alpha = 2.0 / (window + 1)
        out[..., 0] = arr[..., 0]
        for t in range(1, steps):
            out[..., t] = alpha * arr[..., t] + (1 - alpha) * out[..., t - 1]

    else:
        raise ValueError(f"smoothing must be None, 'sma', or 'ema', got {smoothing!r}.")

    return out


def _compute_uncertainty(x: np.ndarray, uncertainty_fun, center: np.ndarray | None = None) -> tuple:
    """
    Compute (lower, upper) bands from x of shape (n, steps).

    Parameters
    ----------
    x : np.ndarray, shape (n, steps)
        The trajectories to compute spread from.
    uncertainty_fun : str | callable | None
    center : np.ndarray, shape (steps,), optional
        Precomputed center line. Used to anchor std/mad bands so they are
        consistent with the plotted aggregate line. If None, recomputed
        from x (mean for std, median for mad).
    """
    if uncertainty_fun is None:
        return None, None

    if callable(uncertainty_fun):
        result = uncertainty_fun(x)
        if len(result) == 2:
            return result
        raise ValueError("Custom uncertainty_fun must return (lower, upper).")

    if uncertainty_fun == "ci95":
        return np.percentile(x, 2.5, axis=0), np.percentile(x, 97.5, axis=0)

    if uncertainty_fun == "std":
        c  = center if center is not None else x.mean(axis=0)
        sd = x.std(axis=0)
        return c - sd, c + sd

    if uncertainty_fun == "mad":
        c          = center if center is not None else np.median(x, axis=0)
        mad        = np.median(np.abs(x - c), axis=0)
        scaled_mad = 1.4826 * mad
        return c - scaled_mad, c + scaled_mad

    raise ValueError(f"uncertainty_fun must be None, 'ci95', 'std', 'mad', or callable. Got {uncertainty_fun!r}.")


def _aggregate_center(x: np.ndarray, aggregate_fun) -> np.ndarray:
    """x: (n, steps) -> (steps,)"""
    if callable(aggregate_fun):
        return np.asarray(aggregate_fun(x))
    if aggregate_fun == "mean":
        return x.mean(axis=0)
    if aggregate_fun == "median":
        return np.median(x, axis=0)
    raise ValueError(f"aggregate_fun must be 'mean', 'median', or callable. Got {aggregate_fun!r}.")


def plot_time_varying_posterior(
    samples: dict,
    local_keys: list,
    aggregate_fun: str | Callable | None = "mean",
    aggregate_strategy: str = "full_uncertainty",
    uncertainty_fun: str | Callable | None = "ci95",
    smoothing: str | None = None,
    smoothing_window: int = 5,
    spaghetti: bool = False,
    marginal: bool = True,
    data_idx: list | None = None,
    num_datasets: int | None = None,
    color: str = "#822621",
    n_cols: int = 2,
    title_fontsize: int = 16,
    label_fontsize: int = 14,
    tick_fontsize: int = 12,
    alpha: float = 0.4,
):
    """
    Plot time-varying parameter posteriors.

    Parameters
    ----------
    samples : dict
        Each value has shape (num_datasets, num_post_samples, num_steps, 1).
    local_keys : list of str
        Which keys to plot.
    aggregate_fun : {"mean", "median"} | callable | None
        Aggregation function over trajectories. If None, no aggregation —
        each selected dataset is shown individually.
    aggregate_strategy : {"full_uncertainty", "no_epistemic"}
        How to prepare trajectories before aggregating.
        - "full_uncertainty": flatten (num_datasets, num_post_samples) together,
          then aggregate. Captures both epistemic and aleatoric uncertainty.
        - "no_epistemic": take per-dataset median across posterior samples first,
          then aggregate across datasets. Removes epistemic uncertainty.
        Ignored when aggregate_fun is None.
    uncertainty_fun : {"ci95", "std", "mad"} | callable | None
        Band to draw around the aggregate line. None draws no band.
        Ignored when aggregate_fun is None.
    smoothing : {None, "sma", "ema"}
        Causal (past-only) smoothing applied to trajectories before plotting.
    smoothing_window : int
        Window length for SMA or EMA span.
    spaghetti : bool
        When aggregate_fun is not None, also draw individual lines behind the
        aggregate. When aggregate_fun is None this has no extra effect.
    marginal : bool
        Whether to draw a marginal KDE panel attached to each trajectory axis.
    data_idx : list of int, optional
        Explicit dataset indices to select.
    num_datasets : int, optional
        Use first n datasets (ignored if data_idx is given).
    """
    # ------------------------------------------------------------------ #
    # dataset selection
    # ------------------------------------------------------------------ #
    num_available = next(iter(samples.values())).shape[0]
    if data_idx is not None:
        idx = list(data_idx)
    elif num_datasets is not None:
        idx = list(range(min(num_datasets, num_available)))
    else:
        idx = list(range(num_available))

    selected = {k: np.asarray(v)[idx] for k, v in samples.items()}
    # each value: (B, S, T, 1) where B = len(idx)

    # ------------------------------------------------------------------ #
    # uncertainty band label
    # ------------------------------------------------------------------ #
    _BAND_LABELS = {
        "ci95": "95% CI",
        "std":  "±1 STD",
        "mad":  "±1.48 MAD",
    }
    if callable(uncertainty_fun):
        band_label = "Uncertainty band"
    else:
        band_label = _BAND_LABELS.get(uncertainty_fun, "Uncertainty band")

    # ------------------------------------------------------------------ #
    # layout
    # ------------------------------------------------------------------ #
    COL_WIDTH, ROW_HEIGHT = 6.0, 3.0
    n_params = len(local_keys)
    B        = len(idx)

    has_legend = True  # always show legend

    if aggregate_fun is None:
        # rows = params, cols = datasets
        n_rows = n_params
        n_cols_fig = B
        fig, axes = plt.subplots(
            n_rows, n_cols_fig,
            figsize=(COL_WIDTH * n_cols_fig, ROW_HEIGHT * n_rows + 0.5),
            squeeze=False,
        )
        # axes shape: (n_params, B)

        for row, name in enumerate(local_keys):
            arr = selected[name][..., 0]   # (B, S, T)
            _, S, T = arr.shape
            t = np.arange(T)

            for col in range(B):
                ax_base = axes[row, col]

                if marginal:
                    sub     = ax_base.get_subplotspec().subgridspec(1, 2, width_ratios=[4.2, 0.8], wspace=0.0)
                    ax_traj = fig.add_subplot(sub[0])
                    ax_kde  = fig.add_subplot(sub[1])
                    ax_base.axis("off")
                else:
                    ax_traj = ax_base
                    ax_kde  = None

                traj = np.median(arr[col], axis=0)            # (T,)
                traj = _apply_smoothing(traj, smoothing, smoothing_window)
                ax_traj.plot(t, traj, color=color, linewidth=1.5, alpha=1.0)

                # title: param name on first row, dataset index on first col
                if row == 0:
                    ax_traj.set_title(f"Dataset {idx[col]}", fontsize=title_fontsize, pad=10)
                if col == 0:
                    ax_traj.set_ylabel(name, fontsize=label_fontsize)
                ax_traj.set_xlabel("Step" if row == n_params - 1 else "", fontsize=label_fontsize, labelpad=8)
                ax_traj.grid(alpha=0.3)
                ax_traj.tick_params(labelsize=tick_fontsize)

                if marginal:
                    kde_vals = _apply_smoothing(arr[col].reshape(-1, T), smoothing, smoothing_window).reshape(-1)
                    sns.kdeplot(y=kde_vals, ax=ax_kde, color=color, fill=True, alpha=1)
                    ax_kde.set_ylim(ax_traj.get_ylim())
                    ax_kde.set_axis_off()

        handles = [
            mlines.Line2D([], [], color=color, linewidth=1.5, label="Posterior median"),
        ]

    else:
        # aggregated: one panel per param
        n_rows = int(np.ceil(n_params / n_cols))
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(COL_WIDTH * n_cols, ROW_HEIGHT * n_rows + 0.5),
        )
        axes = np.atleast_1d(axes).ravel()

        for i, name in enumerate(local_keys):
            ax_base = axes[i]
            arr = selected[name][..., 0]   # (B, S, T)
            B_i, S, T = arr.shape
            t = np.arange(T)

            if marginal:
                sub     = ax_base.get_subplotspec().subgridspec(1, 2, width_ratios=[4.2, 0.8], wspace=0.0)
                ax_traj = fig.add_subplot(sub[0])
                ax_kde  = fig.add_subplot(sub[1])
                ax_base.axis("off")
            else:
                ax_traj = ax_base
                ax_kde  = None

            if aggregate_strategy == "full_uncertainty":
                flat = arr.reshape(B_i * S, T)
            elif aggregate_strategy == "no_epistemic":
                flat = np.median(arr, axis=1)
            else:
                raise ValueError(
                    f"aggregate_strategy must be 'full_uncertainty' or 'no_epistemic', "
                    f"got {aggregate_strategy!r}."
                )

            flat = _apply_smoothing(flat, smoothing, smoothing_window)

            if spaghetti:
                per_ds = _apply_smoothing(np.median(arr, axis=1), smoothing, smoothing_window)
                for d in range(B_i):
                    ax_traj.plot(t, per_ds[d], color=color, linewidth=1.0, alpha=alpha)

            center = _aggregate_center(flat, aggregate_fun)
            lower, upper = _compute_uncertainty(flat, uncertainty_fun, center=center)

            if lower is not None:
                ax_traj.fill_between(t, lower, upper, color=color, alpha=0.25, edgecolor="none", zorder=1)
            ax_traj.plot(t, center, color=color, linewidth=2.5, zorder=3)

            ax_traj.set_title(name, fontsize=title_fontsize, pad=10)
            ax_traj.set_xlabel("Step", fontsize=label_fontsize, labelpad=8)
            ax_traj.grid(alpha=0.3)
            ax_traj.tick_params(labelsize=tick_fontsize)

            if marginal:
                sns.kdeplot(y=flat.reshape(-1), ax=ax_kde, color=color, fill=True, alpha=1)
                ax_kde.set_ylim(ax_traj.get_ylim())
                ax_kde.set_axis_off()

        for j in range(n_params, len(axes)):
            axes[j].axis("off")

        handles = [
            mlines.Line2D([], [], color=color, linewidth=2.5, label="Aggregate"),
        ]
        if uncertainty_fun is not None:
            handles.append(mpatches.Patch(color=color, alpha=0.25, label=band_label))
        if spaghetti:
            handles.append(mlines.Line2D([], [], color=color, linewidth=1.0, alpha=1, label="Individual"))

    # ------------------------------------------------------------------ #
    # shared legend
    # ------------------------------------------------------------------ #
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        fontsize=label_fontsize,
        framealpha=0.0,
        bbox_to_anchor=(0.5, -0.02),
    )

    sns.despine()
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    return fig


# def plot_time_invariant_posterior(
#     samples: dict,
#     hyper_keys: list,
#     shared_keys: list,
#     mixture_names: dict | None = None,
#     aggregate: bool = True,
#     data_idx: list | None = None,
#     num_datasets: int | None = None,
#     color: str = "#822621",
#     num_cols: int = 2,
#     title_fontsize: int = 16,
#     tick_fontsize: int = 12,
#     alpha: float = 0.8,
# ):
#     """
#     Plot time-invariant parameter posteriors.

#     Parameters
#     ----------
#     samples : dict
#         Output of approximator.sample().
#         Each value has shape (num_datasets, num_samples, num_steps, 1).
#     hyper_keys : list of str
#     shared_keys : list of str
#     mixture_names : dict, optional
#         {param_name: [component_name, ...]}
#     aggregate : bool
#         If True, pool all datasets. If False, overlay per-dataset KDEs.
#     data_idx : list of int, optional
#     num_datasets : int, optional
#     """
#     selected = _select_datasets(samples, aggregate, data_idx, num_datasets)

#     labeled_params = {}
#     for k in hyper_keys:
#         labeled_params[f"{k}  [hyper]"] = selected[k]
#     for k in shared_keys:
#         labeled_params[f"{k}  [shared]"] = selected[k]

#     if not labeled_params:
#         raise ValueError("No time-invariant parameters to plot.")

#     COL_WIDTH, ROW_HEIGHT = 5.0, 3.0
#     n      = len(labeled_params)
#     n_rows = int(np.ceil(n / num_cols))

#     fig, axes = plt.subplots(
#         n_rows, num_cols,
#         figsize=(COL_WIDTH * num_cols, ROW_HEIGHT * n_rows),
#     )
#     axes = np.atleast_1d(axes).ravel()

#     for i, (label, arr) in enumerate(labeled_params.items()):
#         ax  = axes[i]
#         # (num_datasets, num_samples, num_steps, dim) -> take first step
#         arr = np.asarray(arr)[:, :, 0, :]  # (num_datasets, num_samples, dim)
#         dim = arr.shape[-1]

#         if dim > 1:
#             param_name  = label.split("_mixture_weights")[0].strip()
#             comp_names  = (
#                 mixture_names.get(param_name) if mixture_names else None
#             ) or [f"component {k}" for k in range(dim)]

#             for k in range(dim):
#                 vals = arr[:, :, k].reshape(-1)
#                 sns.kdeplot(
#                     vals, ax=ax,
#                     color=PALETTE[k % len(PALETTE)],
#                     fill=True, alpha=alpha, linewidth=2.0,
#                     label=comp_names[k],
#                 )
#             ax.legend(fontsize=tick_fontsize, framealpha=0.3)

#         else:
#             if aggregate:
#                 sns.kdeplot(
#                     arr[:, :, 0].reshape(-1),
#                     ax=ax, color=color,
#                     fill=True, alpha=alpha, linewidth=2.0,
#                 )
#             else:
#                 n_ds        = arr.shape[0]
#                 ds_colors   = _trajectory_palette(color, n_ds)
#                 for d in range(n_ds):
#                     sns.kdeplot(
#                         arr[d, :, 0],
#                         ax=ax, color=ds_colors[d],
#                         fill=True, alpha=alpha, linewidth=1.5,
#                     )

#         ax.set_title(label, fontsize=title_fontsize, pad=10)
#         ax.set_xlabel("")
#         ax.set_ylabel("")
#         ax.grid(alpha=0.3)
#         ax.tick_params(labelsize=tick_fontsize)

#     for j in range(len(labeled_params), len(axes)):
#         axes[j].axis("off")

#     sns.despine()
#     plt.tight_layout()

#     return fig


# def plot_joint_posterior(
#     samples: dict,
#     local_keys: list,
#     hyper_keys: list,
#     shared_keys: list,
#     mixture_names: dict | None = None,
#     color: str = "#822621",
#     title_fontsize: int = 18,
#     tick_fontsize: int = 12,
#     alpha: float = 0.4,
# ):
#     """
#     Plot joint posterior aggregated across all datasets.
#     Rows = model parameters, cols = hyper param KDEs + trajectory.

#     Parameters
#     ----------
#     samples : dict
#         Output of approximator.sample().
#         Each value has shape (num_datasets, num_samples, num_steps, 1).
#     local_keys : list of str
#     hyper_keys : list of str
#     shared_keys : list of str
#     mixture_names : dict, optional
#     """
#     all_param_names = list(dict.fromkeys(
#         local_keys +
#         shared_keys +
#         [k.split("_")[0] for k in hyper_keys]
#     ))

#     row_specs = []
#     for param_name in all_param_names:
#         h_cols = [
#             (k, np.asarray(samples[k])[:, :, 0, :])  # (num_datasets, num_samples, dim)
#             for k in hyper_keys
#             if k.startswith(param_name + "_")
#         ]
#         # (num_datasets, num_samples, num_steps)
#         local_arr  = np.asarray(samples[param_name])[..., 0]  if param_name in local_keys  else None
#         # (num_datasets, num_samples)
#         shared_arr = np.asarray(samples[param_name])[:, :, 0, 0] if param_name in shared_keys else None

#         row_specs.append({
#             "name":       param_name,
#             "hyper_cols": h_cols,
#             "local":      local_arr,
#             "shared":     shared_arr,
#         })

#     if not row_specs:
#         raise ValueError("No parameters found.")

#     max_hyper = max(len(r["hyper_cols"]) for r in row_specs)
#     n_cols    = max_hyper + 1
#     n_rows    = len(row_specs)

#     num_trajectories = min(10, next(
#         (r["local"].shape[0] for r in row_specs if r["local"] is not None), 1
#     ))
#     traj_colors = _trajectory_palette(color, num_trajectories)

#     COL_WIDTH, ROW_HEIGHT = 4.0, 3.0
#     fig = plt.figure(figsize=(COL_WIDTH * n_cols, ROW_HEIGHT * n_rows))

#     col_widths = [1.0] * (n_cols - 1) + [2.0]
#     gs = gridspec.GridSpec(n_rows, n_cols, width_ratios=col_widths, figure=fig)
#     axes = np.array([
#         [fig.add_subplot(gs[r, c]) for c in range(n_cols)]
#         for r in range(n_rows)
#     ])

#     for row_i, spec in enumerate(row_specs):
#         param_name = spec["name"]
#         hyper_cols = spec["hyper_cols"]
#         local_arr  = spec["local"]   # (num_datasets, num_samples, num_steps)
#         shared_arr = spec["shared"]  # (num_datasets, num_samples)

#         # -- hyper kde columns --
#         for col_i, (label, arr) in enumerate(hyper_cols):
#             ax  = axes[row_i, col_i]
#             dim = arr.shape[-1]

#             if dim > 1:
#                 comp_names = (
#                     mixture_names.get(param_name) if mixture_names else None
#                 ) or [f"component {k}" for k in range(dim)]

#                 for k in range(dim):
#                     sns.kdeplot(
#                         arr[:, :, k].reshape(-1),
#                         ax=ax,
#                         color=PALETTE[k % len(PALETTE)],
#                         fill=True, alpha=0.8, linewidth=2.0,
#                         label=comp_names[k],
#                     )
#                 ax.legend(fontsize=tick_fontsize, framealpha=0.3)
#             else:
#                 sns.kdeplot(
#                     arr[:, :, 0].reshape(-1),
#                     ax=ax, color=color,
#                     fill=True, alpha=0.8, linewidth=2.0,
#                 )

#             short_label = "_".join(label.split("_")[1:])
#             ax.set_title(short_label, fontsize=title_fontsize, pad=15)
#             ax.set_xlabel("")
#             ax.set_ylabel("")
#             ax.grid(alpha=0.3)
#             ax.tick_params(labelsize=tick_fontsize)

#         # -- shared param --
#         if shared_arr is not None:
#             ax = axes[row_i, 0]
#             sns.kdeplot(
#                 shared_arr.reshape(-1),
#                 ax=ax, color=color,
#                 fill=True, alpha=0.8, linewidth=2.0,
#             )
#             ax.set_xlabel("")
#             ax.set_ylabel("")
#             ax.grid(alpha=0.3)
#             ax.tick_params(labelsize=tick_fontsize)

#         # -- trajectory column --
#         ax_traj = axes[row_i, n_cols - 1]
#         if local_arr is not None:
#             t      = np.arange(local_arr.shape[-1])
#             n_plot = min(num_trajectories, local_arr.shape[0])

#             for d in range(n_plot):
#                 traj = np.median(local_arr[d], axis=0)
#                 ax_traj.plot(t, traj, color=traj_colors[d], linewidth=2, alpha=alpha)

#             mean_traj = np.median(local_arr.reshape(-1, local_arr.shape[-1]), axis=0)
#             ax_traj.plot(t, mean_traj, color="black", linewidth=2.5, alpha=1.0)

#             ax_traj.set_title("Trajectory", fontsize=title_fontsize, pad=15)
#             ax_traj.set_xlabel("")
#             ax_traj.grid(alpha=0.3)
#             ax_traj.tick_params(labelsize=tick_fontsize)
#         else:
#             ax_traj.axis("off")

#         # -- blank unused cols --
#         for col_i in range(len(hyper_cols), n_cols - 1):
#             if shared_arr is None or col_i > 0:
#                 axes[row_i, col_i].axis("off")

#     plt.tight_layout()
#     plt.draw()

#     for row_i, spec in enumerate(row_specs):
#         ax0  = axes[row_i, 0]
#         bbox = ax0.get_position()
#         fig.text(
#             0.01, bbox.y0 + bbox.height / 2,
#             spec["name"],
#             ha="center", va="center",
#             fontsize=title_fontsize,
#             rotation=0,
#         )

#     fig.subplots_adjust(left=0.06)
#     sns.despine()

#     return fig