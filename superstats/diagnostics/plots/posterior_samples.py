import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import seaborn as sns


PALETTE = ["#822621", "#C1440E", "#E8871A", "#D4A843", ]


def _trajectory_palette(base_color: str, n: int) -> list:
    warm_ramp = ["#D4A843", base_color, "#822621"]
    cmap = mcolors.LinearSegmentedColormap.from_list("warm_traj", warm_ramp)
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def _select_datasets(
    samples: dict,
    aggregate: bool,
    data_idx: list | None,
    num_datasets: int | None,
) -> dict:
    """Slice samples to selected datasets or return all for aggregation."""
    if aggregate:
        return samples

    num_available = next(iter(samples.values())).shape[0]

    if data_idx is not None:
        idx = data_idx
    elif num_datasets is not None:
        idx = list(range(min(num_datasets, num_available)))
    else:
        idx = list(range(num_available))

    return {k: v[idx] for k, v in samples.items()}


def plot_time_varying_posterior(
    samples: dict,
    local_keys: list,
    aggregate: bool = True,
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
        Output of approximator.sample().
        Each value has shape (num_datasets, num_samples, num_steps, 1).
    local_keys : list of str
        Which keys to plot.
    aggregate : bool
        If True, pool all datasets into a single mean + 95% CI.
        If False, plot median trajectory per dataset.
    data_idx : list of int, optional
        Explicit dataset indices. Only used when aggregate=False.
    num_datasets : int, optional
        Plot first n datasets. Only used when aggregate=False.
    """
    selected = _select_datasets(samples, aggregate, data_idx, num_datasets)

    COL_WIDTH, ROW_HEIGHT = 5.0, 3.0
    n      = len(local_keys)
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(COL_WIDTH * n_cols, ROW_HEIGHT * n_rows),
    )
    axes = np.atleast_1d(axes).ravel()

    for i, name in enumerate(local_keys):
        ax = axes[i]
        # (num_datasets, num_samples, num_steps, 1) -> (num_datasets, num_samples, num_steps)
        arr = np.asarray(selected[name])[..., 0]

        sub     = ax.get_subplotspec().subgridspec(1, 2, width_ratios=[4.2, 0.8], wspace=0.0)
        ax_traj = fig.add_subplot(sub[0])
        ax_kde  = fig.add_subplot(sub[1])

        t = np.arange(arr.shape[-1])

        if aggregate:
            # pool across datasets and samples -> (num_datasets * num_samples, num_steps)
            flat  = arr.reshape(-1, arr.shape[-1])
            mean  = flat.mean(axis=0)
            lower = np.percentile(flat, 2.5,  axis=0)
            upper = np.percentile(flat, 97.5, axis=0)

            ax_traj.plot(t, mean, color=color, linewidth=2.5)
            ax_traj.fill_between(t, lower, upper, color=color, alpha=0.25)
            kde_vals = flat.reshape(-1)
        else:
            n_ds        = arr.shape[0]
            traj_colors = _trajectory_palette(color, n_ds)

            for d in range(n_ds):
                traj = np.median(arr[d], axis=0)
                ax_traj.plot(t, traj, color=traj_colors[d], linewidth=1.5, alpha=alpha)

            mean_traj = np.median(arr.reshape(-1, arr.shape[-1]), axis=0)
            ax_traj.plot(t, mean_traj, color="black", linewidth=2.5, alpha=1.0)
            kde_vals = arr.reshape(-1)

        ax_traj.set_title(name, fontsize=title_fontsize, pad=10)
        ax_traj.set_xlabel("step", fontsize=label_fontsize)
        ax_traj.grid(alpha=0.3)
        ax_traj.tick_params(labelsize=tick_fontsize)

        sns.kdeplot(y=kde_vals, ax=ax_kde, color=color, fill=True, alpha=0.8)
        ax_kde.set_ylim(ax_traj.get_ylim())
        ax_kde.set_axis_off()
        ax.axis("off")

    for j in range(len(local_keys), len(axes)):
        axes[j].axis("off")

    sns.despine()
    plt.tight_layout()

    return fig


def plot_time_invariant_posterior(
    samples: dict,
    hyper_keys: list,
    shared_keys: list,
    mixture_names: dict | None = None,
    aggregate: bool = True,
    data_idx: list | None = None,
    num_datasets: int | None = None,
    color: str = "#822621",
    num_cols: int = 2,
    title_fontsize: int = 16,
    tick_fontsize: int = 12,
    alpha: float = 0.8,
):
    """
    Plot time-invariant parameter posteriors.

    Parameters
    ----------
    samples : dict
        Output of approximator.sample().
        Each value has shape (num_datasets, num_samples, num_steps, 1).
    hyper_keys : list of str
    shared_keys : list of str
    mixture_names : dict, optional
        {param_name: [component_name, ...]}
    aggregate : bool
        If True, pool all datasets. If False, overlay per-dataset KDEs.
    data_idx : list of int, optional
    num_datasets : int, optional
    """
    selected = _select_datasets(samples, aggregate, data_idx, num_datasets)

    labeled_params = {}
    for k in hyper_keys:
        labeled_params[f"{k}  [hyper]"] = selected[k]
    for k in shared_keys:
        labeled_params[f"{k}  [shared]"] = selected[k]

    if not labeled_params:
        raise ValueError("No time-invariant parameters to plot.")

    COL_WIDTH, ROW_HEIGHT = 5.0, 3.0
    n      = len(labeled_params)
    n_rows = int(np.ceil(n / num_cols))

    fig, axes = plt.subplots(
        n_rows, num_cols,
        figsize=(COL_WIDTH * num_cols, ROW_HEIGHT * n_rows),
    )
    axes = np.atleast_1d(axes).ravel()

    for i, (label, arr) in enumerate(labeled_params.items()):
        ax  = axes[i]
        # (num_datasets, num_samples, num_steps, dim) -> take first step
        arr = np.asarray(arr)[:, :, 0, :]  # (num_datasets, num_samples, dim)
        dim = arr.shape[-1]

        if dim > 1:
            param_name  = label.split("_mixture_weights")[0].strip()
            comp_names  = (
                mixture_names.get(param_name) if mixture_names else None
            ) or [f"component {k}" for k in range(dim)]

            for k in range(dim):
                vals = arr[:, :, k].reshape(-1)
                sns.kdeplot(
                    vals, ax=ax,
                    color=PALETTE[k % len(PALETTE)],
                    fill=True, alpha=alpha, linewidth=2.0,
                    label=comp_names[k],
                )
            ax.legend(fontsize=tick_fontsize, framealpha=0.3)

        else:
            if aggregate:
                sns.kdeplot(
                    arr[:, :, 0].reshape(-1),
                    ax=ax, color=color,
                    fill=True, alpha=alpha, linewidth=2.0,
                )
            else:
                n_ds        = arr.shape[0]
                ds_colors   = _trajectory_palette(color, n_ds)
                for d in range(n_ds):
                    sns.kdeplot(
                        arr[d, :, 0],
                        ax=ax, color=ds_colors[d],
                        fill=True, alpha=alpha, linewidth=1.5,
                    )

        ax.set_title(label, fontsize=title_fontsize, pad=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=tick_fontsize)

    for j in range(len(labeled_params), len(axes)):
        axes[j].axis("off")

    sns.despine()
    plt.tight_layout()

    return fig


def plot_joint_posterior(
    samples: dict,
    local_keys: list,
    hyper_keys: list,
    shared_keys: list,
    mixture_names: dict | None = None,
    color: str = "#822621",
    title_fontsize: int = 18,
    tick_fontsize: int = 12,
    alpha: float = 0.4,
):
    """
    Plot joint posterior aggregated across all datasets.
    Rows = model parameters, cols = hyper param KDEs + trajectory.

    Parameters
    ----------
    samples : dict
        Output of approximator.sample().
        Each value has shape (num_datasets, num_samples, num_steps, 1).
    local_keys : list of str
    hyper_keys : list of str
    shared_keys : list of str
    mixture_names : dict, optional
    """
    all_param_names = list(dict.fromkeys(
        local_keys +
        shared_keys +
        [k.split("_")[0] for k in hyper_keys]
    ))

    row_specs = []
    for param_name in all_param_names:
        h_cols = [
            (k, np.asarray(samples[k])[:, :, 0, :])  # (num_datasets, num_samples, dim)
            for k in hyper_keys
            if k.startswith(param_name + "_")
        ]
        # (num_datasets, num_samples, num_steps)
        local_arr  = np.asarray(samples[param_name])[..., 0]  if param_name in local_keys  else None
        # (num_datasets, num_samples)
        shared_arr = np.asarray(samples[param_name])[:, :, 0, 0] if param_name in shared_keys else None

        row_specs.append({
            "name":       param_name,
            "hyper_cols": h_cols,
            "local":      local_arr,
            "shared":     shared_arr,
        })

    if not row_specs:
        raise ValueError("No parameters found.")

    max_hyper = max(len(r["hyper_cols"]) for r in row_specs)
    n_cols    = max_hyper + 1
    n_rows    = len(row_specs)

    num_trajectories = min(10, next(
        (r["local"].shape[0] for r in row_specs if r["local"] is not None), 1
    ))
    traj_colors = _trajectory_palette(color, num_trajectories)

    COL_WIDTH, ROW_HEIGHT = 4.0, 3.0
    fig = plt.figure(figsize=(COL_WIDTH * n_cols, ROW_HEIGHT * n_rows))

    col_widths = [1.0] * (n_cols - 1) + [2.0]
    gs = gridspec.GridSpec(n_rows, n_cols, width_ratios=col_widths, figure=fig)
    axes = np.array([
        [fig.add_subplot(gs[r, c]) for c in range(n_cols)]
        for r in range(n_rows)
    ])

    for row_i, spec in enumerate(row_specs):
        param_name = spec["name"]
        hyper_cols = spec["hyper_cols"]
        local_arr  = spec["local"]   # (num_datasets, num_samples, num_steps)
        shared_arr = spec["shared"]  # (num_datasets, num_samples)

        # -- hyper kde columns --
        for col_i, (label, arr) in enumerate(hyper_cols):
            ax  = axes[row_i, col_i]
            dim = arr.shape[-1]

            if dim > 1:
                comp_names = (
                    mixture_names.get(param_name) if mixture_names else None
                ) or [f"component {k}" for k in range(dim)]

                for k in range(dim):
                    sns.kdeplot(
                        arr[:, :, k].reshape(-1),
                        ax=ax,
                        color=PALETTE[k % len(PALETTE)],
                        fill=True, alpha=0.8, linewidth=2.0,
                        label=comp_names[k],
                    )
                ax.legend(fontsize=tick_fontsize, framealpha=0.3)
            else:
                sns.kdeplot(
                    arr[:, :, 0].reshape(-1),
                    ax=ax, color=color,
                    fill=True, alpha=0.8, linewidth=2.0,
                )

            short_label = "_".join(label.split("_")[1:])
            ax.set_title(short_label, fontsize=title_fontsize, pad=15)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=tick_fontsize)

        # -- shared param --
        if shared_arr is not None:
            ax = axes[row_i, 0]
            sns.kdeplot(
                shared_arr.reshape(-1),
                ax=ax, color=color,
                fill=True, alpha=0.8, linewidth=2.0,
            )
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=tick_fontsize)

        # -- trajectory column --
        ax_traj = axes[row_i, n_cols - 1]
        if local_arr is not None:
            t      = np.arange(local_arr.shape[-1])
            n_plot = min(num_trajectories, local_arr.shape[0])

            for d in range(n_plot):
                traj = np.median(local_arr[d], axis=0)
                ax_traj.plot(t, traj, color=traj_colors[d], linewidth=2, alpha=alpha)

            mean_traj = np.median(local_arr.reshape(-1, local_arr.shape[-1]), axis=0)
            ax_traj.plot(t, mean_traj, color="black", linewidth=2.5, alpha=1.0)

            ax_traj.set_title("Trajectory", fontsize=title_fontsize, pad=15)
            ax_traj.set_xlabel("")
            ax_traj.grid(alpha=0.3)
            ax_traj.tick_params(labelsize=tick_fontsize)
        else:
            ax_traj.axis("off")

        # -- blank unused cols --
        for col_i in range(len(hyper_cols), n_cols - 1):
            if shared_arr is None or col_i > 0:
                axes[row_i, col_i].axis("off")

    plt.tight_layout()
    plt.draw()

    for row_i, spec in enumerate(row_specs):
        ax0  = axes[row_i, 0]
        bbox = ax0.get_position()
        fig.text(
            0.01, bbox.y0 + bbox.height / 2,
            spec["name"],
            ha="center", va="center",
            fontsize=title_fontsize,
            rotation=0,
        )

    fig.subplots_adjust(left=0.06)
    sns.despine()

    return fig