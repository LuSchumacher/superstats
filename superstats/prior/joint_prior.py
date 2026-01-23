from typing import Dict, Union, Any
import numpy as np
from typing import Union, Dict, Any


class JointPrior:
    def __init__(self, **kwargs: "Union['Transition', 'Prior']"):
        self.params = kwargs

    def sample(self, steps: int) -> Dict[str, Dict[str, Any]]:
        """
        Sample all parameters.

        Returns a dict with keys:
        - 'local_params': dict param_name -> sampled trajectories from Transition.sample(steps)
        - 'global_params': dict param_name -> transition hyperparameters (e.g. sigma)
        - 'shared_params': dict param_name -> samples from fixed Priors (size=1)
        """
        from superstats.transition.transition import Transition
        from superstats.prior.prior import Prior
        local_params = {}
        global_params = {}
        shared_params = {}

        for name, param in self.params.items():
            if isinstance(param, Transition):
                samples = param.sample(steps)
                local_params[name] = samples["local_param"]
                for k, v in samples.items():
                    if k != "local_param":
                        global_params[f"{name}_{k}"] = v
            elif isinstance(param, Prior):
                shared_params[name] = param.sample(size=1).item()
            else:
                raise TypeError(f"Unknown parameter type for {name}: {type(param)}")

        return {
            "local_params": local_params,
            "global_params": global_params,
            "shared_params": shared_params,
        }