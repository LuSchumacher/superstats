"""Joint priors over time-varying and time-invariant parameters."""

from typing import Any, Dict

import numpy as np

from superstats.diagnostics.plots.prior_samples import (
    plot_joint_prior,
    plot_time_invariant_prior,
    plot_time_varying_prior,
)
from .prior import Prior
from superstats.transition import DeterministicTransition, StochasticTransition


class JointPrior:
    """Joint prior over multiple model parameters.

    Parameters
    ----------
    **kwargs : StochasticTransition, DeterministicTransition, Prior, float, int
        Named model parameters.

        Use `StochasticTransition` for stochastic time-varying parameters with
        hyperparameters, `DeterministicTransition` for deterministic
        time-varying parameters, `Prior` for inferred time-invariant
        parameters, and scalar values for fixed parameters.

    Notes
    -----
    Sample outputs are grouped into:

    - `local_params`: stochastic time-varying parameters (inferred).
    - `deterministic_params`: deterministic time-varying parameters (no inferred).
    - `hyper_params`: hyperparameters for transition models (inferred).
    - `shared_params`: time-invariant parameters (inferred).
    - `fixed_params`: fixed parameters (no inferred).
    """

    def __init__(self, **kwargs: StochasticTransition | DeterministicTransition | Prior | float | int):
        self.params = kwargs
        self._last_hyper_param_groups = {}

    def sample(self, batch_size: int, num_steps: int) -> Dict[str, Any]:
        """Draw a joint parameter sample.

        Parameters
        ----------
        batch_size : int
            Number of independent samples to draw.
        num_steps  : int
            Number of time steps per trajectory.

        Returns
        -------
        result : dict - sampled parameter groups `local_params`,
            `deterministic_params` `hyper_params`, `shared_params`,
            and `fixed_params`.

        Raises
        ------
        ValueError
            If batch_size or num_steps is not a positive integer.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if num_steps <= 0:
            raise ValueError("num_steps must be a positive integer")

        local_params = {}
        deterministic_params = {}
        hyper_params = {}
        shared_params = {}
        fixed_params = {}
        hyper_param_groups = {}

        for name, param in self.params.items():
            if isinstance(param, (StochasticTransition, DeterministicTransition)):
                samples = param.sample(batch_size=batch_size, num_steps=num_steps)
                target = deterministic_params if isinstance(param, DeterministicTransition) else local_params
                sample_key = "deterministic_params" if isinstance(param, DeterministicTransition) else "local_params"
                target[name] = samples[sample_key]

                hyper_param_groups[name] = []
                for key, value in samples["hyper_params"].items():
                    full_key = f"{name}_{key}"
                    hyper_params[full_key] = value
                    hyper_param_groups[name].append(full_key)

                for key, value in samples["fixed_params"].items():
                    fixed_params[f"{name}_{key}"] = value

            elif isinstance(param, Prior):
                shared_params[name] = param.sample(batch_size=batch_size)

            elif np.isscalar(param):
                fixed_params[name] = int(param) if isinstance(param, int) else float(param)

            else:
                raise TypeError(f"Unknown parameter type for '{name}': {type(param).__name__}")

        self._last_hyper_param_groups = hyper_param_groups
        return {
            "local_params": local_params,
            "deterministic_params": deterministic_params,
            "hyper_params": hyper_params,
            "shared_params": shared_params,
            "fixed_params": fixed_params,
        }

    def _param_bounds(self) -> dict:
        """Collect y-axis bounds declared on the underlying parameter objects.

        Returns
        -------
        bounds : dict - mapping from parameter name to its `bounds`
            attribute, for parameters that define one
        """
        return {
            name: obj.bounds for name, obj in self.params.items() if hasattr(obj, "bounds") and obj.bounds is not None
        }

    def _mixture_names(self) -> dict:
        """Collect mixture component names declared on the underlying parameter objects.

        Returns
        -------
        names : dict - mapping from parameter name to its `names`
            attribute, for parameters that define one
        """
        return {name: obj.names for name, obj in self.params.items() if hasattr(obj, "names")}

    def plot_time_varying_prior(self, num_steps: int = 200, num_trajectories: int = 20, **kwargs):
        """Plot sampled time-varying prior trajectories.

        Parameters
        ----------
        num_steps        : int, optional, default: 200
            Number of time steps to sample per trajectory.
        num_trajectories : int, optional, default: 20
            Number of trajectories to draw.
        **kwargs         : dict, optional, default: {}
            Further optional keyword arguments propagated to the
            underlying `plot_time_varying_prior` plotting function.

        Returns
        -------
        fig : plt.Figure - the generated figure
        """
        samples = self.sample(batch_size=num_trajectories, num_steps=num_steps)
        local_params = {**samples["local_params"], **samples["deterministic_params"]}
        return plot_time_varying_prior(
            local_params=local_params,
            param_bounds=self._param_bounds(),
            **kwargs,
        )

    def plot_time_invariant_prior(self, num_draws: int = 1000, **kwargs):
        """Plot marginal distributions for time-invariant prior parameters.

        Parameters
        ----------
        num_draws : int, optional, default: 1000
            Number of draws used to sample `hyper_params` and `shared_params`.
        **kwargs  : dict, optional, default: {}
            Further optional keyword arguments propagated to the
            underlying `plot_time_invariant_prior` plotting function.

        Returns
        -------
        fig : plt.Figure - the generated figure
        """
        samples = self.sample(batch_size=num_draws, num_steps=1)
        return plot_time_invariant_prior(
            hyper_params=samples["hyper_params"],
            shared_params=samples["shared_params"],
            mixture_names=self._mixture_names(),
            **kwargs,
        )

    def plot_joint_prior(self, num_steps: int = 200, num_trajectories: int = 20, num_draws: int = 1000, **kwargs):
        """Plot joint prior diagnostics across local and shared parameters.

        Parameters
        ----------
        num_steps        : int, optional, default: 200
            Number of time steps for local trajectory sampling.
        num_trajectories : int, optional, default: 20
            Number of local trajectories to plot.
        num_draws        : int, optional, default: 1000
            Number of draws used for time-invariant parameter sampling.
        **kwargs         : dict, optional, default: {}
            Further optional keyword arguments propagated to the
            underlying `plot_joint_prior` plotting function.

        Returns
        -------
        fig : plt.Figure - the generated figure
        """
        samples = self.sample(batch_size=num_draws, num_steps=num_steps)
        all_local_params = {**samples["local_params"], **samples["deterministic_params"]}
        local_params = {k: v[:num_trajectories] for k, v in all_local_params.items()}
        return plot_joint_prior(
            local_params=local_params,
            hyper_params=samples["hyper_params"],
            shared_params=samples["shared_params"],
            param_bounds=self._param_bounds(),
            mixture_names=self._mixture_names(),
            hyper_param_groups=self._last_hyper_param_groups,
            **kwargs,
        )
