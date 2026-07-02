from typing import Any, Dict

import numpy as np

from superstats.diagnostics.plots.prior_samples import (
    plot_joint_prior,
    plot_time_invariant_prior,
    plot_time_varying_prior,
)
from superstats.prior.prior import Prior
from superstats.transition.transition import Transition


class JointPrior:
    """Joint prior over multiple model parameters.

    Parameters
    ----------
    **kwargs : Transition, Prior, float, int
        Named model parameters. Use ``Transition`` for time-varying
        parameters with hyperparameters, ``Prior`` for inferred stationary
        parameters, and scalar values for fixed parameters.

    Notes
    -----
    Sample outputs are grouped into:

    - ``local_params``: time-varying trajectories.
    - ``hyper_params``: inferred transition hyperparameters.
    - ``shared_params``: inferred stationary parameters.
    - ``fixed_params``: fixed scalar values.
    """

    def __init__(self, **kwargs: Transition | Prior | float | int):
        self.params = kwargs

    def sample(self, batch_size: int, num_steps: int) -> Dict[str, Any]:
        """Draw a joint parameter sample.

        Parameters
        ----------
        batch_size : int
            Number of independent samples to draw.
        num_steps : int
            Number of time steps per trajectory.

        Returns
        -------
        dict
            Sampled parameter groups: ``local_params``, ``hyper_params``,
            ``shared_params``, and ``fixed_params``.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if num_steps <= 0:
            raise ValueError("num_steps must be a positive integer")

        local_params: Dict[str, np.ndarray] = {}
        hyper_params: Dict[str, np.ndarray] = {}
        shared_params: Dict[str, np.ndarray] = {}
        fixed_params: Dict[str, Any] = {}

        for name, param in self.params.items():
            if isinstance(param, Transition):
                samples = param.sample(batch_size=batch_size, num_steps=num_steps)
                local_params[name] = samples["local_params"]

                for key, value in samples["hyper_params"].items():
                    hyper_params[f"{name}_{key}"] = value

                for key, value in samples["fixed_params"].items():
                    fixed_params[f"{name}_{key}"] = value

            elif isinstance(param, Prior):
                shared_params[name] = param.sample(batch_size=batch_size)

            elif np.isscalar(param):
                fixed_params[name] = int(param) if isinstance(param, int) else float(param)

            else:
                raise TypeError(
                    f"Unknown parameter type for '{name}': {type(param).__name__}"
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
        """Plot sampled time-varying prior trajectories.

        Parameters
        ----------
        num_steps : int, optional
            Number of time steps to sample per trajectory.
        num_trajectories : int, optional
            Number of trajectories to draw.
        **kwargs
            Passed through to ``plot_time_varying_prior``.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        samples = self.sample(batch_size=num_trajectories, num_steps=num_steps)
        return plot_time_varying_prior(
            local_params=samples["local_params"],
            param_bounds=self._param_bounds(),
            **kwargs,
        )

    def plot_time_invariant_prior(self, num_draws: int = 1000, **kwargs):
        """Plot marginal distributions for time-invariant prior parameters.

        Parameters
        ----------
        num_draws : int, optional
            Number of draws used to sample ``hyper_params`` and
            ``shared_params``.
        **kwargs
            Passed through to ``plot_time_invariant_prior``.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        samples = self.sample(batch_size=num_draws, num_steps=1)
        return plot_time_invariant_prior(
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
        """Plot joint prior diagnostics across local and shared parameters.

        Parameters
        ----------
        num_steps : int, optional
            Number of time steps for local trajectory sampling.
        num_trajectories : int, optional
            Number of local trajectories to plot.
        num_draws : int, optional
            Number of draws used for time-invariant parameter sampling.
        **kwargs
            Passed through to ``plot_joint_prior``.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        samples = self.sample(batch_size=num_draws, num_steps=num_steps)
        local_params = {
            k: v[:num_trajectories]
            for k, v in samples["local_params"].items()
        }
        return plot_joint_prior(
            local_params=local_params,
            hyper_params=samples["hyper_params"],
            shared_params=samples["shared_params"],
            param_bounds=self._param_bounds(),
            mixture_names=self._mixture_names(),
            **kwargs,
        )