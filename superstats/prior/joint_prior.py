from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import seaborn as sns

from superstats.transition.transition import Transition
from superstats.prior.prior import Prior
from superstats.diagnostics.plots.prior_samples import (
    plot_time_varying_prior  as _plot_time_varying_prior,
    plot_time_invariant_prior as _plot_time_invariant_prior,
    plot_joint_prior          as _plot_joint_prior,
)

PALETTE = ["#822621", "#C1440E", "#E8871A", "#D4A843"]


class JointPrior:
    """
    Joint prior over multiple parameters.

    Supports:
    - Transition -> time-varying parameters (with hyperparameters)
    - Prior -> inferred shared parameters
    - float/int -> fixed parameters

    Returns structured output:
    - local_params  : time-varying (batch, num_steps)
    - hyper_params  : inferred hyperparameters
    - shared_params : inferred stationary parameters (batch,)
    - fixed_params  : all fixed values (including fixed hyperparameters)
    """

    def __init__(self, **kwargs: Transition | Prior | float | int):
        self.params = kwargs

    def sample(self, batch_size: int, num_steps: int) -> Dict[str, Any]:

        local_params: Dict[str, np.ndarray] = {}
        hyper_params: Dict[str, np.ndarray] = {}
        shared_params: Dict[str, np.ndarray] = {}
        fixed_params: Dict[str, np.ndarray] = {}

        for name, param in self.params.items():
            if isinstance(param, Transition):
                samples = param.sample(batch_size=batch_size, num_steps=num_steps)
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

    def _param_bounds(self) -> dict:
        return {
            name: obj.bounds
            for name, obj in self.params.items()
            if hasattr(obj, "bounds") and obj.bounds is not None
        }

    def _mixture_names(self) -> dict:
        return {
            name: obj.names
            for name, obj in self.params.items()
            if hasattr(obj, "names")
        }

    def plot_time_varying_prior(
        self,
        num_steps: int = 200,
        num_trajectories: int = 20,
        **kwargs
    ):
        samples = self.sample(batch_size=num_trajectories, num_steps=num_steps)
        return _plot_time_varying_prior(
            local_params=samples["local_params"],
            param_bounds=self._param_bounds(),
            **kwargs,
        )

    def plot_time_invariant_prior(self, num_draws: int = 1000, **kwargs):
        samples = self.sample(batch_size=num_draws, num_steps=1)
        return _plot_time_invariant_prior(
            hyper_params=samples["hyper_params"],
            shared_params=samples["shared_params"],
            mixture_names=self._mixture_names(),
            **kwargs,
        )

    def plot_joint_prior(
        self,
        num_steps: int = 200,
        num_trajectories: int = 20,
        num_draws: int = 2000,
        **kwargs
    ):
        samples = self.sample(batch_size=num_draws, num_steps=num_steps)
        local_params = {
            k: v[:num_trajectories]
            for k, v in samples["local_params"].items()
        }
        return _plot_joint_prior(
            local_params=local_params,
            hyper_params=samples["hyper_params"],
            shared_params=samples["shared_params"],
            param_bounds=self._param_bounds(),
            mixture_names=self._mixture_names(),
            **kwargs,
        )