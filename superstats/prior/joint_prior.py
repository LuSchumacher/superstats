"""Joint priors over time-varying and time-invariant parameters."""

from typing import Any, Dict

import numpy as np
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
        self._last_fixed_param_groups = {}

    def sample(
        self,
        batch_size: int,
        num_steps: int,
    ) -> Dict[str, Any]:
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
        fixed_param_groups = {}

        for name, param in self.params.items():
            if isinstance(param, StochasticTransition):
                target = local_params
                sample_key = "local_params"
            elif isinstance(param, DeterministicTransition):
                target = deterministic_params
                sample_key = "deterministic_params"
            elif isinstance(param, Prior):
                shared_params[name] = param.sample(batch_size=batch_size)
                continue
            elif np.isscalar(param):
                fixed_params[name] = param
                continue
            else:
                raise TypeError(f"Unknown parameter type for '{name}': {type(param).__name__}")

            sample_kwargs = {"batch_size": batch_size, "num_steps": num_steps}
            samples = param.sample(**sample_kwargs)
            target[name] = samples[sample_key]

            hyper_param_groups[name] = []
            for key, value in samples["hyper_params"].items():
                full_key = f"{name}_{key}"
                hyper_params[full_key] = value
                hyper_param_groups[name].append(full_key)

            fixed_param_groups[name] = []
            for key, value in samples["fixed_params"].items():
                full_key = f"{name}_{key}"
                fixed_params[full_key] = value
                fixed_param_groups[name].append(full_key)

        self._last_hyper_param_groups = hyper_param_groups
        self._last_fixed_param_groups = fixed_param_groups
        return {
            "local_params": local_params,
            "deterministic_params": deterministic_params,
            "hyper_params": hyper_params,
            "shared_params": shared_params,
            "fixed_params": fixed_params,
        }
