from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from superstats.transition.transition import Transition
from superstats.prior.prior import Prior


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
            # Transition (time-varying)
            if isinstance(param, Transition):
                samples = param.sample(batch_size=batch_size, steps=steps)
                local_params[name] = samples["local_params"]

                # hyperparameters
                for k, v in samples["hyper_params"].items():
                    full_name = f"{k}_{name}"
                    hyper_params[full_name] = v

                # fixed hyperparameters
                for k, v in samples["fixed_params"].items():
                    full_name = f"{k}_{name}"
                    fixed_params[full_name] = v

            # ------------------------
            # Prior (shared, inferred)
            # ------------------------
            elif isinstance(param, Prior):
                values = param.sample(batch_size=batch_size)
                shared_params[name] = values

            # ------------------------
            # Fixed scalar
            # ------------------------
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

    # --------------------------------------------------
    # Plotting (updated to new structure)
    # --------------------------------------------------
    def plot_prior(
        self,
        steps: int = 100,
        num_trajectories: int = 10,
        num_draws: int = 1000,
        color: str = "#822621",
        n_cols: int = 2,
        title_fontsize: int = 14,
        label_fontsize: int = 11,
        tick_fontsize: int = 9,
    ):

        samples = self.sample(batch_size=num_draws, steps=steps)

        local = samples["local_params"]
        shared = samples["shared_params"]
        hyper = samples["hyper_params"]

        sections = []

        if local:
            sections.append(("Local parameters", local, "line"))
        if shared:
            sections.append(("Shared parameters", shared, "hist"))
        if hyper:
            sections.append(("Hyper parameters", hyper, "hist"))

        if not sections:
            raise ValueError("No parameters to plot.")

        COL_WIDTH = 5.0
        ROW_HEIGHT = 3.0

        for section_title, params, kind in sections:

            n = len(params)
            n_rows = int(np.ceil(n / n_cols))

            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(COL_WIDTH * n_cols, ROW_HEIGHT * n_rows),
            )

            axes = np.atleast_1d(axes).ravel()
            fig.suptitle(section_title, fontsize=title_fontsize)

            i = 0

            for name, values in params.items():
                ax = axes[i]

                # ------------------------
                # trajectories
                # ------------------------
                if kind == "line":
                    n_plot = min(num_trajectories, values.shape[0])
                    values_plot = np.asarray(values[:n_plot])

                    sub = ax.get_subplotspec().subgridspec(
                        1, 2,
                        width_ratios=[4.2, 0.8],
                        wspace=0.0
                    )

                    ax_traj = fig.add_subplot(sub[0])
                    ax_kde = fig.add_subplot(sub[1])

                    param_obj = self.params.get(name, None)
                    if hasattr(param_obj, "bounds") and param_obj.bounds is not None:
                        ax_traj.set_ylim(param_obj.bounds)

                    for j in range(n_plot):
                        ax_traj.plot(values_plot[j], alpha=0.6, color=color)

                    ax_traj.set_xlabel("step", fontsize=label_fontsize)
                    ax_traj.set_ylabel(name, fontsize=label_fontsize)
                    ax_traj.grid(alpha=0.3)
                    ax_traj.tick_params(labelsize=tick_fontsize)

                    kde_values = values_plot.reshape(-1)

                    sns.kdeplot(
                        y=kde_values,
                        ax=ax_kde,
                        color=color,
                        fill=True,
                        alpha=0.4,
                    )

                    ax_kde.set_ylim(ax_traj.get_ylim())
                    ax_kde.set_axis_off()
                    ax.axis("off")

                # ------------------------
                # distributions
                # ------------------------
                else:
                    sns.histplot(
                        values,
                        bins=30,
                        stat="density",
                        kde=True,
                        ax=ax,
                        color=color,
                    )

                    ax.set_xlabel(name, fontsize=label_fontsize)
                    ax.grid(alpha=0.3)
                    ax.tick_params(labelsize=tick_fontsize)

                i += 1

            for j in range(i, len(axes)):
                axes[j].axis("off")

            sns.despine()
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()