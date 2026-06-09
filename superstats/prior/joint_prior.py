from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import seaborn as sns

from superstats.transition.transition import Transition
from superstats.prior.prior import Prior

PALETTE = ["#C1440E", "#E8871A", "#D4A843", "#7B3F00"]


class JointPrior:
    """
    Joint prior over multiple parameters.

    Supports:
    - Transition -> time-varying parameters (with hyperparameters)
    - Prior -> inferred shared parameters
    - float/int -> fixed parameters

    Returns structured output:
    - local_params  : time-varying (batch, steps)
    - hyper_params  : inferred hyperparameters
    - shared_params : inferred stationary parameters (batch,)
    - fixed_params  : all fixed values (including fixed hyperparameters)
    """

    def __init__(self, **kwargs: Transition | Prior | float | int):
        self.params = kwargs

    def sample(self, batch_size: int, steps: int) -> Dict[str, Any]:

        local_params: Dict[str, np.ndarray] = {}
        hyper_params: Dict[str, np.ndarray] = {}
        shared_params: Dict[str, np.ndarray] = {}
        fixed_params: Dict[str, np.ndarray] = {}

        for name, param in self.params.items():
            if isinstance(param, Transition):
                samples = param.sample(batch_size=batch_size, steps=steps)
                local_params[name] = samples["local_params"]

                for k, v in samples["hyper_params"].items():
                    hyper_params[f"{name}_{k}"] = v

                for k, v in samples["fixed_params"].items():
                    fixed_params[f"{name}_{k}"] = v
            # Shared parameters
            elif isinstance(param, Prior):
                values = param.sample(batch_size=batch_size)
                shared_params[name] = values
            # Fixed parameters
            elif np.isscalar(param):
                fixed_params[name] = float(param) if not isinstance(param, int) else int(param)
            else:
                raise TypeError(
                    f"Unknown parameter type for '{name}': {type(param)}"
                )

        return {
            "local_params": local_params,
            "hyper_params": hyper_params,
            "shared_params": shared_params,
            "fixed_params": fixed_params,
        }

    def plot_time_varying_prior(
        self,
        steps: int = 100,
        num_trajectories: int = 2,
        color: str = "#822621",
        n_cols: int = 2,
        title_fontsize: int = 16,
        label_fontsize: int = 14,
        tick_fontsize: int = 12,
        alpha: float = 0.8,
    ):
        samples = self.sample(batch_size=num_trajectories, steps=steps)
        local = samples["local_params"]

        if not local:
            raise ValueError("No time-varying (local) parameters to plot.")

        COL_WIDTH, ROW_HEIGHT = 5.0, 3.0
        n = len(local)
        n_rows = int(np.ceil(n / n_cols))

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(COL_WIDTH * n_cols, ROW_HEIGHT * n_rows),
        )
        axes = np.atleast_1d(axes).ravel()

        for i, (name, values) in enumerate(local.items()):
            ax = axes[i]
            n_plot = min(num_trajectories, values.shape[0])
            values_plot = np.asarray(values[:n_plot])

            sub = ax.get_subplotspec().subgridspec(1, 2, width_ratios=[4.2, 0.8], wspace=0.0)
            ax_traj = fig.add_subplot(sub[0])
            ax_kde = fig.add_subplot(sub[1])

            param_obj = self.params.get(name)
            if hasattr(param_obj, "bounds") and param_obj.bounds is not None:
                ax_traj.set_ylim(param_obj.bounds)

            for j in range(n_plot):
                ax_traj.plot(values_plot[j], alpha=alpha, color=color)

            ax_traj.set_title(name, fontsize=title_fontsize, pad=10)
            ax_traj.set_xlabel("step", fontsize=label_fontsize)
            ax_traj.grid(alpha=0.3)
            ax_traj.tick_params(labelsize=tick_fontsize)

            sns.kdeplot(y=values_plot.reshape(-1), ax=ax_kde, color=color, fill=True, alpha=alpha)
            ax_kde.set_ylim(ax_traj.get_ylim())
            ax_kde.set_axis_off()
            ax.axis("off")

        for j in range(len(local), len(axes)):
            axes[j].axis("off")

        sns.despine()
        plt.tight_layout()
        
        return fig
    
    def plot_time_invariant_prior(
        self,
        num_draws: int = 1000,
        color: str = "#822621",
        num_cols: int = 2,
        title_fontsize: int = 16,
        tick_fontsize: int = 12,
        alpha: float = 0.8,
    ):
        samples = self.sample(batch_size=num_draws, steps=1)
        hyper = samples["hyper_params"]
        shared = samples["shared_params"]

        if not hyper and not shared:
            raise ValueError("No time-invariant parameters to plot.")

        labeled_params = {}
        for name, values in hyper.items():
            labeled_params[f"{name}  [hyper]"] = values
        for name, values in shared.items():
            labeled_params[f"{name}  [shared]"] = values

        COL_WIDTH, ROW_HEIGHT = 5.0, 3.0
        n = len(labeled_params)
        n_rows = int(np.ceil(n / num_cols))

        fig, axes = plt.subplots(
            n_rows, num_cols,
            figsize=(COL_WIDTH * num_cols, ROW_HEIGHT * n_rows),
        )
        axes = np.atleast_1d(axes).ravel()

        for i, (label, values) in enumerate(labeled_params.items()):
            ax = axes[i]
            arr = np.asarray(values)

            if arr.ndim == 2 and arr.shape[1] > 1:
                param_name = label.split("_mixture_weights")[0].strip()
                mixture_obj = self.params.get(param_name)

                if hasattr(mixture_obj, "names") and len(mixture_obj.names) == arr.shape[1]:
                    component_names = mixture_obj.names
                else:
                    component_names = [f"component {k}" for k in range(arr.shape[1])]

                for k in range(arr.shape[1]):
                    sns.histplot(
                        arr[:, k],
                        bins=30,
                        stat="density",
                        kde=True,
                        line_kws={"linewidth": 3.0},
                        ax=ax,
                        color=PALETTE[k % len(PALETTE)],
                        alpha=alpha,
                        label=component_names[k],
                    )
                ax.legend(fontsize=tick_fontsize, framealpha=0.3)
            else:
                sns.histplot(
                    arr.reshape(-1),
                    bins=30,
                    stat="density",
                    kde=True,
                    line_kws={"linewidth": 3.0},
                    ax=ax,
                    color=color,
                    alpha=alpha,
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
    
    @staticmethod
    def _trajectory_palette(base_color: str, n: int) -> list:
        warm_ramp = ["#C4846A", base_color, "#4A0E0E"]
        cmap = mcolors.LinearSegmentedColormap.from_list("warm_traj", warm_ramp)
        return [cmap(i / max(n - 1, 1)) for i in range(n)]

    def plot_joint_prior(
        self,
        steps: int = 200,
        num_trajectories: int = 5,
        num_draws: int = 2000,
        color: str = "#822621",
        title_fontsize: int = 18,
        tick_fontsize: int = 12,
        alpha: float = 0.8,
    ):
        samples = self.sample(batch_size=max(num_draws, num_trajectories), steps=steps)
        local  = samples["local_params"]
        hyper  = samples["hyper_params"]
        shared = samples["shared_params"]

        row_specs = []

        for param_name, param_obj in self.params.items():
            if isinstance(param_obj, (int, float)) or np.isscalar(param_obj):
                continue

            hyper_cols = [
                (k, np.asarray(v))
                for k, v in hyper.items()
                if k.startswith(param_name + "_")
            ]
            local_arr  = np.asarray(local[param_name])  if param_name in local  else None
            shared_arr = np.asarray(shared[param_name]) if param_name in shared else None

            row_specs.append({
                "name":       param_name,
                "hyper_cols": hyper_cols,
                "local":      local_arr,
                "shared":     shared_arr,
            })

        if not row_specs:
            raise ValueError("No plottable parameters found.")

        max_hyper = max(len(r["hyper_cols"]) for r in row_specs)
        n_cols = max_hyper + 1
        n_rows = len(row_specs)

        traj_colors = self._trajectory_palette(color, num_trajectories)

        COL_WIDTH, ROW_HEIGHT = 4.0, 3.0
        TRAJ_RATIO = 2

        fig = plt.figure(figsize=(
            COL_WIDTH * n_cols,
            ROW_HEIGHT * n_rows,
        ))

        col_widths = [1.0] * (n_cols - 1) + [TRAJ_RATIO]
        gs = gridspec.GridSpec(
            n_rows, n_cols,
            width_ratios=col_widths,
            figure=fig,
        )
        axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(n_cols)] for r in range(n_rows)])

        for row_i, spec in enumerate(row_specs):
            param_name = spec["name"]
            hyper_cols = spec["hyper_cols"]
            local_arr  = spec["local"]
            shared_arr = spec["shared"]

            # -- hyper hist columns --
            for col_i, (label, values) in enumerate(hyper_cols):
                ax = axes[row_i, col_i]
                arr = np.asarray(values)

                if arr.ndim == 2 and arr.shape[1] > 1:
                    param_obj = self.params.get(param_name)
                    if hasattr(param_obj, "names") and len(param_obj.names) == arr.shape[1]:
                        component_names = param_obj.names
                    else:
                        component_names = [f"component {k}" for k in range(arr.shape[1])]

                    for k in range(arr.shape[1]):
                        sns.histplot(
                            arr[:, k],
                            bins=30,
                            stat="density",
                            kde=True,
                            line_kws={"linewidth": 2.0},
                            ax=ax,
                            color=PALETTE[k % len(PALETTE)],
                            alpha=alpha,
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
                        alpha=alpha,
                    )

                short_label = "_".join(label.split("_")[1:])
                ax.set_title(short_label, fontsize=title_fontsize, pad=15)
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.grid(alpha=0.3)
                ax.tick_params(labelsize=tick_fontsize)

            # -- shared param: single hist in col 0 --
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
                    alpha=alpha,
                )
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.grid(alpha=0.3)
                ax.tick_params(labelsize=tick_fontsize)

            # -- trajectory column --
            ax_traj = axes[row_i, n_cols - 1]
            if local_arr is not None:
                n_plot = min(num_trajectories, local_arr.shape[0])
                for j in range(n_plot):
                    ax_traj.plot(
                        local_arr[j],
                        alpha=alpha,
                        color=traj_colors[j],
                        linewidth=2,
                    )

                param_obj = self.params.get(param_name)
                if hasattr(param_obj, "bounds") and param_obj.bounds is not None:
                    ax_traj.set_ylim(param_obj.bounds)

                ax_traj.set_title("Trajectory", fontsize=title_fontsize, pad=15)
                ax_traj.set_xlabel("")
                ax_traj.grid(alpha=0.3)
                ax_traj.tick_params(labelsize=tick_fontsize)
            else:
                ax_traj.axis("off")

            # -- blank unused hyper columns --
            for col_i in range(len(hyper_cols), n_cols - 1):
                if shared_arr is None or col_i > 0:
                    axes[row_i, col_i].axis("off")

        # -- row labels --
        plt.tight_layout()
        plt.draw()

        for row_i, spec in enumerate(row_specs):
            ax0 = axes[row_i, 0]
            bbox = ax0.get_position()
            row_center_y = bbox.y0 + bbox.height / 2
            fig.text(
                0.01, row_center_y,
                spec["name"],
                ha="center", va="center",
                fontsize=title_fontsize,
                rotation=0,
            )

        fig.subplots_adjust(left=0.06)
        sns.despine()

        return fig