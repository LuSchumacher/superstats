from typing import Dict, Union, Any
import numpy as np

from superstats.transition.transition import Transition
from superstats.prior.prior import Prior


class JointPrior:
    """
    A joint prior distribution combining multiple parameter priors and transitions.

    This class manages a collection of parameter distributions that can be either
    time-invariant priors (shared across trajectories) or time-varying transitions
    (evolving through time). It provides batched sampling for efficient simulation.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments mapping parameter names to their distributions.
        Values can be either Prior objects (for time-invariant parameters) or
        Transition objects (for time-varying parameters).
    """

    def __init__(self, **kwargs: "Union['Transition', 'Prior']"):
        """
        Initialize the joint prior with parameter distributions.

        Parameters
        ----------
        **kwargs : dict
            Parameter name to distribution mapping. Each value should be either
            a Prior (for shared parameters) or Transition (for trajectory parameters).
        """
        self.params = kwargs

    def sample(self, batch_size: int, steps: int) -> Dict[str, Dict[str, Any]]:
        """
        Sample from all parameter distributions in a batched manner.

        This method generates samples for all configured parameters, handling both
        time-varying transitions and time-invariant priors. The output is structured
        to separate local trajectory parameters, global hyperparameters, and shared
        parameters for efficient use in generative models.

        Parameters
        ----------
        batch_size : int
            Number of independent samples/trajectories to generate.
        steps : int
            Number of time steps for trajectory parameters.

        Returns
        -------
        dict
            Dictionary with the following keys:
            - 'local_params': dict
                Time-varying parameters, mapping parameter names to arrays of
                shape (batch_size, steps).
            - 'global_params': dict
                Hyperparameters from transition processes, with keys like
                'param_name_hyperparam' and arrays of shape (batch_size,).
            - 'shared_params': dict, optional
                Time-invariant parameters, mapping parameter names to arrays of
                shape (batch_size,). Only present if shared parameters exist.

        Raises
        ------
        TypeError
            If a parameter value is neither a Prior nor Transition object.
        """
        local_params: Dict[str, np.ndarray] = {}
        global_params: Dict[str, np.ndarray] = {}
        shared_params: Dict[str, np.ndarray] = {}

        for name, param in self.params.items():
            if isinstance(param, Transition):
                # Sampling of trajectories
                samples = param.sample(batch_size=batch_size, steps=steps)
                local_params[name] = samples["local_params"]
                # Store hyperparameters in global_params
                for k, v in samples.items():
                    if k != "local_params":
                        global_params[f"{name}_{k}"] = v
            elif isinstance(param, Prior):
                # Sampling of fixed parameters
                shared_params[name] = param.sample(batch_size=batch_size)
            else:
                raise TypeError(f"Unknown parameter type for {name}: {type(param)}")

        result = {
            "local_params": local_params,
            "global_params": global_params,
        }

        if shared_params:
            result["shared_params"] = shared_params

        return result