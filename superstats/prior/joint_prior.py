from typing import Dict, Union, Any
import numpy as np

from superstats.transition.transition import Transition
from superstats.prior.prior import Prior


class JointPrior:
    def __init__(self, **kwargs: "Union['Transition', 'Prior']"):
        """
        kwargs: dict of parameter_name -> Transition or Prior
        """
        self.params = kwargs

    def sample(self, batch_size: int, steps: int) -> Dict[str, Dict[str, Any]]:
        """
        Fully batched sampling of all parameters.

        Returns:
        - local_params: dict param_name -> array (batch_size, steps)
        - global_params: dict param_name -> array (batch_size,) for transition hyperparameters
        - shared_params: dict param_name -> array (batch_size,) for time-invariant parameters
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