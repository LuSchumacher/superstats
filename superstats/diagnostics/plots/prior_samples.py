import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import seaborn as sns

plt.rcParams["axes.axisbelow"] = True

PALETTE = [
    "#822621",
    "#C1440E",
    "#E8871A",
    "#D4A843",
]


def plot_time_varying_prior(
    local_params: dict,
    param_bounds: dict | None = None,
    color: str = "#822621",
    num_cols: int = 2,
    title_fontsize: int = 16,
    label_fontsize: int = 14,
    tick_fontsize: int = 12,
    alpha: float = 0.5,
):
    """
    Plot time-varying parameter trajectories with marginal KDE.

    Parameters
    ----------
    local_params : dict
        Mapping from parameter name to an array of shape
        ``(num_trajectories, num_steps)``.
    param_bounds : dict, optional
        Mapping from parameter name to ``(lower, upper)`` y-axis limits.
    color : str, default "#822621"
        Line color for individual trajectories and KDE.
    num_cols : int, default 2
        Number of subplot columns.
    title_fontsize : int, default 16
    label_fontsize : int, default 14
    tick_fontsize : int, default 12
    alpha : float, default 0.5
        Transparency for individual trajectories.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the plotted trajectories and KDEs.
    """
    if not local_params:
        raise ValueError("No time-varying (local) parameters to plot.")

    COL_WIDTH, ROW_HEIGHT = 6.0, 3.0
    n = len(local_params)
    n_rows = int(np.ceil(n / num_cols))

    fig, axes = plt.subplots(
        n_rows,
        num_cols,
        figsize=(COL_WIDTH * num_cols, ROW_HEIGHT * n_rows + 0.5),
    )
    axes = np.atleast_1d(axes).ravel()

    for i, (name, values) in enumerate(local_params.items()):
        ax = axes[i]
        values_plot = np.asarray(values)
        n_plot = values_plot.shape[0]

        sub = ax.get_subplotspec().subgridspec(
            1, 2, width_ratios=[4.2, 0.8], wspace=0.0
        )
        ax_traj = fig.add_subplot(sub[0])
        ax_kde = fig.add_subplot(sub[1])

        if param_bounds and name in param_bounds:
            ax_traj.set_ylim(param_bounds[name])

        for j in range(n_plot):
            ax_traj.plot(values_plot[j], alpha=alpha, color=color, linewidth=1.5)

        mean_traj = values_plot.mean(axis=0)
        ax_traj.plot(mean_traj, color="black", linewidth=2.5, alpha=1.0)

        ax_traj.set_title(name, fontsize=title_fontsize, pad=10)
        ax_traj.set_xlabel("Step", fontsize=label_fontsize, labelpad=10)
        ax_traj.grid(alpha=0.3)
        ax_traj.tick_params(labelsize=tick_fontsize)

        sns.kdeplot(
            y=values_plot.reshape(-1), ax=ax_kde, color=color, fill=True, alpha=1
        )

        ax_kde.set_ylim(ax_traj.get_ylim())
        ax_kde.set_axis_off()
        ax.axis("off")

    for j in range(len(local_params), len(axes)):
        axes[j].axis("off")

    legend_handles = [
        mlines.Line2D([], [], color="black", linewidth=2.5, label="Average"),
        mlines.Line2D([], [], color=color, linewidth=1.5, alpha=1, label="Individual"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        fontsize=label_fontsize,
        framealpha=0.0,
        bbox_to_anchor=(0.5, -0.05),
    )

    sns.despine()
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    return fig


def plot_time_invariant_prior(
    hyper_params: dict,
    shared_params: dict,
    mixture_names: dict | None = None,
    color: str = "#822621",
    num_cols: int = 2,
    title_fontsize: int = 16,
    tick_fontsize: int = 12,
):
    """
    Plot time-invariant parameter distributions.

    Parameters
    ----------
    hyper_params : dict
        Mapping from parameter name to an array of hyperparameter samples.
    shared_params : dict
        Mapping from parameter name to an array of shared parameter samples.
    mixture_names : dict, optional
        Mapping from parameter name to a list of component names for
        mixture weight parameters.
    color : str, default "#822621"
        Base color for non-mixture histograms.
    num_cols : int, default 2
        Number of subplot columns.
    title_fontsize : int, default 16
    tick_fontsize : int, default 12

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the plotted parameter distributions.
    """
    if not hyper_params and not shared_params:
        raise ValueError("No time-invariant parameters to plot.")

    labeled_params = {}
    for name, values in hyper_params.items():
        labeled_params[f"{name}  [hyper]"] = values
    for name, values in shared_params.items():
        labeled_params[f"{name}  [shared]"] = values

    COL_WIDTH, ROW_HEIGHT = 5.0, 3.0
    n = len(labeled_params)
    n_rows = int(np.ceil(n / num_cols))

    fig, axes = plt.subplots(
        n_rows,
        num_cols,
        figsize=(COL_WIDTH * num_cols, ROW_HEIGHT * n_rows),
    )
    axes = np.atleast_1d(axes).ravel()

    for i, (label, values) in enumerate(labeled_params.items()):
        ax = axes[i]
        arr = np.asarray(values)

        if arr.ndim == 2 and arr.shape[1] > 1:
            param_name = label.split("_mixture_weights")[0].strip()
            component_names = (
                mixture_names.get(param_name) if mixture_names else None
            ) or [f"component {k}" for k in range(arr.shape[1])]

            for k in range(arr.shape[1]):
                sns.histplot(
                    arr[:, k],
                    bins=30,
                    stat="density",
                    kde=True,
                    line_kws={"linewidth": 3.0},
                    ax=ax,
                    color=PALETTE[k % len(PALETTE)],
                    alpha=1,
                    label=component_names[k],
                )
            ax.legend(fontsize=tick_fontsize, framealpha=0.0)
        else:
            sns.histplot(
                arr.reshape(-1),
                bins=30,
                stat="density",
                kde=True,
                line_kws={"linewidth": 3.0},
                ax=ax,
                color=color,
                alpha=1,
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


def plot_joint_prior(
    local_params: dict,
    hyper_params: dict,
    shared_params: dict,
    param_bounds: dict | None = None,
    mixture_names: dict | None = None,
    color: str = "#822621",
    title_fontsize: int = 18,
    tick_fontsize: int = 12,
    alpha: float = 0.5,
):
    """
    Plot joint prior diagnostics combining hyperparameter distributions,
    shared parameter histograms, and time-varying trajectories.

    Parameters
    ----------
    local_params : dict
        Mapping from parameter name to an array of shape
        ``(num_trajectories, num_steps)``.
    hyper_params : dict
        Mapping from hyperparameter name to an array of samples. Keys may
        include parameter prefixes followed by component labels.
    shared_params : dict
        Mapping from parameter name to an array of shared parameter samples.
    param_bounds : dict, optional
        Mapping from parameter name to ``(lower, upper)`` y-axis limits.
    mixture_names : dict, optional
        Mapping from parameter name to a list of component names for mixture
        weight parameters.
    color : str, default "#822621"
        Base plotting color for KDEs and trajectories.
    title_fontsize : int, default 18
    tick_fontsize : int, default 12
    alpha : float, default 0.5
        Transparency for individual trajectories.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the joint prior diagnostics.
    """
    all_param_names = list(
        dict.fromkeys(
            list(local_params.keys())
            + list(shared_params.keys())
            + [k.split("_")[0] for k in hyper_params.keys()]
        )
    )

    row_specs = []
    for param_name in all_param_names:
        hyper_cols = [
            (k, np.asarray(v))
            for k, v in hyper_params.items()
            if k.startswith(param_name + "_")
        ]
        local_arr = (
            np.asarray(local_params[param_name]) if param_name in local_params else None
        )
        shared_arr = (
            np.asarray(shared_params[param_name])
            if param_name in shared_params
            else None
        )

        row_specs.append(
            {
                "name": param_name,
                "hyper_cols": hyper_cols,
                "local": local_arr,
                "shared": shared_arr,
            }
        )

    if not row_specs:
        raise ValueError("No plottable parameters found.")

    max_hyper = max(len(r["hyper_cols"]) for r in row_specs)
    n_cols = max_hyper + 1
    n_rows = len(row_specs)

    COL_WIDTH, ROW_HEIGHT = 4.0, 3.0
    fig = plt.figure(figsize=(COL_WIDTH * n_cols, ROW_HEIGHT * n_rows + 0.5))

    col_widths = [1.0] * (n_cols - 1) + [2.0]
    gs = gridspec.GridSpec(n_rows, n_cols, width_ratios=col_widths, figure=fig)
    axes = np.array(
        [[fig.add_subplot(gs[r, c]) for c in range(n_cols)] for r in range(n_rows)]
    )

    for row_i, spec in enumerate(row_specs):
        param_name = spec["name"]
        hyper_cols = spec["hyper_cols"]
        local_arr = spec["local"]
        shared_arr = spec["shared"]

        for col_i, (label, values) in enumerate(hyper_cols):
            ax = axes[row_i, col_i]
            arr = np.asarray(values)

            if arr.ndim == 2 and arr.shape[1] > 1:
                component_names = (
                    mixture_names.get(param_name) if mixture_names else None
                ) or [f"component {k}" for k in range(arr.shape[1])]

                for k in range(arr.shape[1]):
                    sns.histplot(
                        arr[:, k],
                        bins=30,
                        stat="density",
                        kde=True,
                        line_kws={"linewidth": 2.0},
                        ax=ax,
                        color=PALETTE[k % len(PALETTE)],
                        alpha=1,
                        label=component_names[k],
                    )
                ax.legend(fontsize=tick_fontsize, framealpha=0.3)
            else:
                sns.histplot(
                    arr.reshape(-1),
                    bins=30,
                    stat="density",
                    kde=True,
                    line_kws={"linewidth": 2.0},
                    ax=ax,
                    color=color,
                    alpha=1,
                )

            short_label = "_".join(label.split("_")[1:])
            ax.set_title(short_label, fontsize=title_fontsize, pad=15)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=tick_fontsize)

        if shared_arr is not None:
            ax = axes[row_i, 0]
            sns.histplot(
                shared_arr.reshape(-1),
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

        ax_cell = axes[row_i, n_cols - 1]
        sub = ax_cell.get_subplotspec().subgridspec(
            1, 2, width_ratios=[4.2, 0.8], wspace=0.0
        )
        ax_traj = fig.add_subplot(sub[0])
        ax_kde = fig.add_subplot(sub[1])
        ax_cell.axis("off")

        if local_arr is not None:
            n_plot = local_arr.shape[0]
            for j in range(n_plot):
                ax_traj.plot(local_arr[j], alpha=alpha, color=color, linewidth=2)

            mean_traj = local_arr.mean(axis=0)
            ax_traj.plot(mean_traj, color="black", linewidth=2.5, alpha=1.0)

            if param_bounds and param_name in param_bounds:
                ax_traj.set_ylim(param_bounds[param_name])

            ax_traj.set_title("Trajectory", fontsize=title_fontsize, pad=15)
            ax_traj.set_xlabel("")
            ax_traj.grid(alpha=0.3)
            ax_traj.tick_params(labelsize=tick_fontsize)

            sns.kdeplot(
                y=local_arr.reshape(-1), ax=ax_kde, color=color, fill=True, alpha=1
            )
            ax_kde.set_ylim(ax_traj.get_ylim())
            ax_kde.set_axis_off()
        else:
            ax_traj.axis("off")
            ax_kde.axis("off")

        for col_i in range(len(hyper_cols), n_cols - 1):
            if shared_arr is None or col_i > 0:
                axes[row_i, col_i].axis("off")

    legend_handles = [
        mlines.Line2D([], [], color="black", linewidth=2.5, label="Average"),
        mlines.Line2D([], [], color=color, linewidth=2.0, alpha=1, label="Individual"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        fontsize=title_fontsize - 2,
        framealpha=0.0,
        bbox_to_anchor=(0.5, -0.05),
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.draw()

    for row_i, spec in enumerate(row_specs):
        ax0 = axes[row_i, 0]
        bbox = ax0.get_position()
        fig.text(
            0.01,
            bbox.y0 + bbox.height / 2,
            spec["name"],
            ha="center",
            va="center",
            fontsize=title_fontsize,
            rotation=0,
        )

    fig.subplots_adjust(left=0.06, bottom=0.06)
    sns.despine()

    return fig
