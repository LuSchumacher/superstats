"""Prior sample visualization helpers."""

from collections.abc import Mapping, Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import seaborn as sns

from superstats.defaults import (
    BASE_COLOR,
    CATEGORICAL_PALETTE,
)

plt.rcParams["axes.axisbelow"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Palatino", "Palatino Linotype", "DejaVu Serif"]

BASE_COL_WIDTH = 6.0
BASE_ROW_HEIGHT = 3.0


def plot_time_varying_prior(
    local_params: Mapping[str, np.ndarray],
    param_bounds: Mapping[str, tuple[float, float]] | None = None,
    num_cols: int = 2,
    marginal: bool = True,
    alpha: float = 0.5,
    color: str = BASE_COLOR,
    title_fontsize: int = 22,
    label_fontsize: int = 18,
    tick_fontsize: int = 16,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Plot time-varying parameter trajectories with marginal KDE.

    Parameters
    ----------
    local_params   : dict of np.ndarray, each of shape (num_trajectories, num_steps)
        Mapping from parameter name to an array of trajectories.
    param_bounds   : dict or None, optional, default: None
        Mapping from parameter name to (lower, upper) y-axis limits.
    num_cols       : int, optional, default: 2
        Number of subplot columns.
    marginal       : bool, optional, default: True
        Whether to draw a marginal KDE panel beside each trajectory
        panel.
    alpha          : float in [0, 1], optional, default: 0.5
        The opacity of individual trajectories.
    color          : str, optional, default: BASE_COLOR
        Line color for individual trajectories and marginal KDE.
    title_fontsize : int, optional, default: 22
        The font size of the panel titles.
    label_fontsize : int, optional, default: 18
        The font size of the axis labels.
    tick_fontsize  : int, optional, default: 16
        The font size of the axis tick labels.
    figsize       : tuple of two floats or None, optional, default: None
        Explicit figure size in inches. If None, the default layout size
        is used.


    Returns
    -------
    fig : plt.Figure - the figure instance for optional saving

    Raises
    ------
    ValueError
        If local_params is empty.
    """
    if not local_params:
        raise ValueError("No time-varying (local) parameters to plot.")

    n = len(local_params)
    num_rows = int(np.ceil(n / num_cols))

    default_figsize = (BASE_COL_WIDTH * num_cols, BASE_ROW_HEIGHT * num_rows + 0.5)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize if figsize is not None else default_figsize)
    axes = np.atleast_1d(axes).ravel()

    for i, (name, values) in enumerate(local_params.items()):
        ax = axes[i]
        values_plot = np.asarray(values)
        n_plot = values_plot.shape[0]

        if marginal:
            sub = ax.get_subplotspec().subgridspec(1, 2, width_ratios=[4.2, 0.8], wspace=0.0)
            ax_traj = fig.add_subplot(sub[0])
            ax_kde = fig.add_subplot(sub[1])
            ax.axis("off")
        else:
            ax_traj = ax
            ax_kde = None

        if param_bounds and name in param_bounds:
            ax_traj.set_ylim(param_bounds[name])

        for j in range(n_plot):
            ax_traj.plot(values_plot[j], alpha=alpha, color=color, linewidth=1.5)

        mean_traj = values_plot.mean(axis=0)
        ax_traj.plot(mean_traj, color=color, linewidth=2.5, alpha=1.0)

        ax_traj.set_title(name, fontsize=title_fontsize, pad=15)
        ax_traj.set_xlabel("")
        ax_traj.set_ylabel("")
        if i // num_cols == num_rows - 1:
            ax_traj.set_xlabel("Step", fontsize=label_fontsize, labelpad=10)
        if i % num_cols == 0:
            ax_traj.set_ylabel("Value", fontsize=label_fontsize, labelpad=10)
        ax_traj.grid(alpha=0.3)
        ax_traj.tick_params(labelsize=tick_fontsize)

        if ax_kde is not None:
            sns.kdeplot(y=values_plot.reshape(-1), ax=ax_kde, color=color, fill=True, alpha=1)
            ax_kde.set_ylim(ax_traj.get_ylim())
            ax_kde.set_axis_off()

    for j in range(len(local_params), len(axes)):
        axes[j].axis("off")

    legend_handles = [
        mlines.Line2D([], [], color=color, linewidth=2.5, label="Average"),
        mlines.Line2D([], [], color=color, linewidth=1.0, alpha=1, label="Individual"),
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
    hyper_params: Mapping[str, np.ndarray],
    shared_params: Mapping[str, np.ndarray],
    mixture_names: Mapping[str, Sequence[str]] | None = None,
    color: str = BASE_COLOR,
    num_cols: int = 2,
    title_fontsize: int = 22,
    label_fontsize: int = 18,
    tick_fontsize: int = 16,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Plot time-invariant parameter distributions.

    Parameters
    ----------
    hyper_params   : dict of np.ndarray
        Mapping from parameter name to an array of hyperparameter samples.
    shared_params  : dict of np.ndarray
        Mapping from parameter name to an array of shared parameter samples.
    mixture_names  : dict or None, optional, default: None
        Mapping from parameter name to a list of component names for
        mixture weight parameters.
    color          : str, optional, default: BASE_COLOR
        Base color for non-mixture histograms.
    num_cols       : int, optional, default: 2
        Number of subplot columns.
    title_fontsize : int, optional, default: 22
        The font size of the panel titles.
    label_fontsize : int, optional, default: 18
        The font size of the axis labels.
    tick_fontsize  : int, optional, default: 16
        The font size of the axis tick labels.
    figsize       : tuple of two floats or None, optional, default: None
        Explicit figure size in inches. If None, the default layout size
        is used.

    Returns
    -------
    fig : plt.Figure - the figure instance for optional saving

    Raises
    ------
    ValueError
        If both hyper_params and shared_params are empty.
    """
    if not hyper_params and not shared_params:
        raise ValueError("No time-invariant parameters to plot.")

    labeled_params = {}
    for name, values in hyper_params.items():
        labeled_params[f"{name}  [hyper]"] = values
    for name, values in shared_params.items():
        labeled_params[f"{name}  [shared]"] = values

    n = len(labeled_params)
    num_rows = int(np.ceil(n / num_cols))

    default_figsize = (BASE_COL_WIDTH * num_cols, BASE_ROW_HEIGHT * num_rows)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize if figsize is not None else default_figsize)
    axes = np.atleast_1d(axes).ravel()

    for i, (label, values) in enumerate(labeled_params.items()):
        ax = axes[i]
        arr = np.asarray(values)

        if arr.ndim == 2 and arr.shape[1] > 1:
            param_name = label.split("_mixture_weights")[0].strip()
            component_names = (mixture_names.get(param_name) if mixture_names else None) or [
                f"component {k}" for k in range(arr.shape[1])
            ]

            for k in range(arr.shape[1]):
                sns.histplot(
                    arr[:, k],
                    bins=30,
                    stat="density",
                    kde=True,
                    line_kws={"linewidth": 3.0},
                    ax=ax,
                    color=CATEGORICAL_PALETTE[k % len(CATEGORICAL_PALETTE)],
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
        if i // num_cols == num_rows - 1:
            ax.set_xlabel("Value", fontsize=label_fontsize, labelpad=10)
        if i % num_cols == 0:
            ax.set_ylabel("Density", fontsize=label_fontsize, labelpad=10)

        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=tick_fontsize)

    for j in range(len(labeled_params), len(axes)):
        axes[j].axis("off")

    sns.despine()
    plt.tight_layout()

    return fig


def plot_joint_prior(
    local_params: Mapping[str, np.ndarray],
    hyper_params: Mapping[str, np.ndarray],
    shared_params: Mapping[str, np.ndarray],
    param_bounds: Mapping[str, tuple[float, float]] | None = None,
    mixture_names: Mapping[str, Sequence[str]] | None = None,
    hyper_param_groups: Mapping[str, Sequence[str]] | None = None,
    marginal: bool = True,
    color: str = BASE_COLOR,
    title_fontsize: int = 22,
    tick_fontsize: int = 16,
    alpha: float = 0.5,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Plot joint prior diagnostics combining hyperparameter distributions,
    shared parameter histograms, and time-varying trajectories.

    Parameters
    ----------
    local_params   : dict of np.ndarray, each of shape (num_trajectories, num_steps)
        Mapping from parameter name to an array of trajectories. Every
        `StochasticTransition` parameter is unconditionally added here by
        `JointPrior.sample`, so together with `shared_params` this is the
        authoritative set of row names.
    hyper_params   : dict of np.ndarray
        Mapping from hyperparameter name to an array of samples. Keys are
        `f"{param_name}_{hyper_key}"`.
    shared_params  : dict of np.ndarray
        Mapping from parameter name to an array of shared parameter samples.
    param_bounds   : dict or None, optional, default: None
        Mapping from parameter name to (lower, upper) y-axis limits.
    mixture_names  : dict or None, optional, default: None
        Mapping from parameter name to a list of component names for
        mixture weight parameters.
    hyper_param_groups : dict or None, optional, default: None
        Mapping from each parameter name to the exact list of
        `hyper_params` keys it owns. Required to correctly separate rows
        when one parameter name is a prefix of another at an underscore
        boundary (e.g. "v_1" and "v_1_2"), since `str.startswith` cannot
        disambiguate that case from key strings alone. When provided
        (e.g. by `JointPrior.plot_joint_prior`), this is used instead of
        prefix matching. If omitted, falls back to prefix matching,
        which can misassign hyperparameters in the presence of such
        name collisions.
    marginal       : bool, optional, default: True
        Whether to draw a marginal KDE panel beside each trajectory
        panel.
    color          : str, optional, default: BASE_COLOR
        Base plotting color for KDEs and trajectories.
    title_fontsize : int, optional, default: 22
        The font size of the panel titles (parameter names).
    tick_fontsize  : int, optional, default: 16
        The font size of the axis tick labels.
    alpha          : float in [0, 1], optional, default: 0.5
        The opacity of individual trajectories.
    figsize       : tuple of two floats or None, optional, default: None
        Explicit figure size in inches. If None, the default layout size
        is used.

    Returns
    -------
    fig : plt.Figure - the figure instance for optional saving

    Raises
    ------
    ValueError
        If no plottable parameters are found across local_params,
        hyper_params, and shared_params.
    """
    all_param_names = list(dict.fromkeys(list(local_params.keys()) + list(shared_params.keys())))

    row_specs = []
    for param_name in all_param_names:
        if hyper_param_groups is not None:
            owned_keys = hyper_param_groups.get(param_name, [])
            hyper_cols = [(k, np.asarray(hyper_params[k])) for k in owned_keys if k in hyper_params]
        else:
            hyper_cols = [(k, np.asarray(v)) for k, v in hyper_params.items() if k.startswith(param_name + "_")]
        local_arr = np.asarray(local_params[param_name]) if param_name in local_params else None
        shared_arr = np.asarray(shared_params[param_name]) if param_name in shared_params else None

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
    num_cols = max_hyper + 1
    num_rows = len(row_specs)

    default_figsize = (4.0 * num_cols, BASE_ROW_HEIGHT * num_rows + 0.5)
    fig = plt.figure(figsize=figsize if figsize is not None else default_figsize)

    col_widths = [1.0] * (num_cols - 1) + [2.0]
    gs = gridspec.GridSpec(num_rows, num_cols, width_ratios=col_widths, figure=fig)
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(num_cols)] for r in range(num_rows)])

    for row_i, spec in enumerate(row_specs):
        param_name = spec["name"]
        hyper_cols = spec["hyper_cols"]
        local_arr = spec["local"]
        shared_arr = spec["shared"]
        prefix = param_name + "_"

        for col_i, (label, values) in enumerate(hyper_cols):
            ax = axes[row_i, col_i]
            arr = np.asarray(values)

            if arr.ndim == 2 and arr.shape[1] > 1:
                component_names = (mixture_names.get(param_name) if mixture_names else None) or [
                    f"component {k}" for k in range(arr.shape[1])
                ]

                for k in range(arr.shape[1]):
                    sns.histplot(
                        arr[:, k],
                        bins=30,
                        stat="density",
                        kde=True,
                        line_kws={"linewidth": 2.0},
                        ax=ax,
                        color=CATEGORICAL_PALETTE[k % len(CATEGORICAL_PALETTE)],
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
                    line_kws={"linewidth": 2.0},
                    ax=ax,
                    color=color,
                    alpha=1,
                )

            short_label = label[len(prefix) :] if label.startswith(prefix) else label
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

        ax_cell = axes[row_i, num_cols - 1]

        if marginal:
            sub = ax_cell.get_subplotspec().subgridspec(1, 2, width_ratios=[4.2, 0.8], wspace=0.0)
            ax_traj = fig.add_subplot(sub[0])
            ax_kde = fig.add_subplot(sub[1])
            ax_cell.axis("off")
        else:
            ax_traj = ax_cell
            ax_kde = None

        if local_arr is not None:
            n_plot = local_arr.shape[0]
            for j in range(n_plot):
                ax_traj.plot(local_arr[j], alpha=alpha, color=color, linewidth=1.5)

            mean_traj = local_arr.mean(axis=0)
            ax_traj.plot(mean_traj, color=color, linewidth=2.5, alpha=1.0)

            if param_bounds and param_name in param_bounds:
                ax_traj.set_ylim(param_bounds[param_name])

            ax_traj.set_title("Trajectory", fontsize=title_fontsize, pad=15)
            ax_traj.set_xlabel("")
            ax_traj.grid(alpha=0.3)
            ax_traj.tick_params(labelsize=tick_fontsize)

            if ax_kde is not None:
                sns.kdeplot(y=local_arr.reshape(-1), ax=ax_kde, color=color, fill=True, alpha=1)
                ax_kde.set_ylim(ax_traj.get_ylim())
                ax_kde.set_axis_off()
        else:
            ax_traj.axis("off")
            if ax_kde is not None:
                ax_kde.axis("off")

        for col_i in range(len(hyper_cols), num_cols - 1):
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
        bbox_to_anchor=(0.5, -0.1),
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.draw()

    for row_i, spec in enumerate(row_specs):
        ax0 = axes[row_i, 0]
        bbox = ax0.get_position()
        fig.text(
            -0.02,
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
