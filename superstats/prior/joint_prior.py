from typing import Dict, Union, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from superstats.transition.transition import Transition
from superstats.prior.prior import Prior


class JointPrior:
    """
    Joint prior over multiple parameters (time-varying + shared).

    Supports:
    - Transition -> time-varying parameters
    - Prior -> inferred shared parameters
    - float/int -> fixed shared parameters (not inferred)

    Attributes
    ----------
    params : dict
        Dictionary mapping parameter names to Transition, Prior, or constant values.
    """

    def __init__(self, **kwargs: Union[Transition, Prior, float, int]):
        """
        Initialize a joint prior over mixed parameter types.

        Parameters
        ----------
        **kwargs
            Named parameters, each either a Transition, Prior, or constant value.
        """
        self.params = kwargs

    def _split_inferred(
        self,
        samples: Dict[str, Any]
    ) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:

        local_params = samples.get("local_params", {})
        global_params = samples.get("global_params", {})
        shared_params = samples.get("shared_params", {})
        infer_mask = samples.get("infer_mask", {})

        # local params are always inferred
        inferred_local = local_params

        # shared parameters
        inferred_shared = {
            name: values
            for name, values in shared_params.items()
            if infer_mask.get(name, False)
        }

        # global parameters
        inferred_global = {
            name: values
            for name, values in global_params.items()
            if infer_mask.get(name, False)
        }

        return inferred_local, inferred_shared, inferred_global

    def sample(self, batch_size: int, steps: int) -> Dict[str, Any]:
        """
        Sample from all priors and transitions.

        Parameters
        ----------
        batch_size : int
            Number of independent samples.
        steps : int
            Time steps for Transition parameters.

        Returns
        -------
        dict
            Contains 'local_params' (trajectories), 'global_params' (hyperparameters),
            'shared_params' (static parameters, if any), and 'infer_mask' (which
            hyperparameters are stochastic, if any).

        Raises
        ------
        TypeError
            If a parameter is neither Transition, Prior, nor scalar.
        """
        local_params: Dict[str, np.ndarray] = {}
        global_params: Dict[str, np.ndarray] = {}
        shared_params: Dict[str, np.ndarray] = {}
        infer_mask: Dict[str, bool] = {}

        for name, param in self.params.items():
            # Transition (time-varying)
            if isinstance(param, Transition):
                samples = param.sample(batch_size=batch_size, steps=steps)

                # trajectories
                local_params[name] = samples["local_params"]

                # global hyperparameters
                for k, v in samples["global_params"].items():
                    global_params[f"{k}_{name}"] = v

                # inference mask for globals
                if "infer_mask" in samples:
                    for k, v in samples["infer_mask"].items():
                        infer_mask[f"{k}_{name}"] = v

            # Prior (shared, inferred)
            elif isinstance(param, Prior):
                values = param.sample(batch_size=batch_size)
                shared_params[name] = values
                infer_mask[name] = True

            # Fixed scalar (shared, not inferred)
            elif np.isscalar(param):
                values = np.full(batch_size, param, dtype=np.float32)
                shared_params[name] = values
                infer_mask[name] = False

            else:
                raise TypeError(
                    f"Unknown parameter type for '{name}': {type(param)}"
                )

        result = {
            "local_params": local_params,
            "global_params": global_params,
        }

        if shared_params:
            result["shared_params"] = shared_params

        if infer_mask:
            result["infer_mask"] = infer_mask

        return result

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
        inferred_local, inferred_shared, inferred_global = self._split_inferred(samples)

        sections = []

        if inferred_local:
            sections.append(("Local parameters", inferred_local, "line"))
        if inferred_shared:
            sections.append(("Shared parameters", inferred_shared, "hist"))
        if inferred_global:
            sections.append(("Global parameters", inferred_global, "hist"))

        if not sections:
            raise ValueError("No inferred parameters to plot.")

        COL_WIDTH = 5.0
        ROW_HEIGHT = 3.0

        for section_title, params, kind in sections:

            n = len(params)

            if kind == "line":
                n_cols = 2
            else:
                n_cols = 2

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
                
                # ----- trajectories -----
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

                # ----- distributions -----
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

                ax.tick_params(labelsize=tick_fontsize)
                ax.grid(alpha=0.3)
                i += 1

            for j in range(i, len(axes)):
                axes[j].axis("off")

            sns.despine()
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()