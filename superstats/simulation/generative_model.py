from typing import Callable, Dict
import inspect
import numpy as np
from numba import njit, prange

from superstats.prior.joint_prior import JointPrior

class GenerativeModel:
    def __init__(
        self,
        prior: JointPrior,
        model: Callable
    ):
        self.prior = prior
        self.model = model
        # Inspect simulator signature
        self.signature = inspect.signature(model)
        self.param_order = list(self.signature.parameters.keys())

    def _prepare_flat_params(self, combined_params, batch_size, steps):
        """
        Broadcast parameters to (batch, steps) and flatten to (batch*steps,)
        """
        flat_params = {}
        for name in self.param_order:
            if name not in combined_params:
                param = self.signature.parameters[name]
                if param.default is inspect.Parameter.empty:
                    raise ValueError(
                        f"Parameter '{name}' required by model but missing in prior."
                    )
                else:
                    # skip parameters with defaults
                    continue
            p = combined_params[name]
            # shared parameters (batch,) → (batch, steps)
            if p.ndim == 1:
                p = np.broadcast_to(p[:, None], (batch_size, steps))
            # already trajectory parameters
            elif p.ndim != 2:
                raise ValueError(
                    f"Unexpected shape for parameter '{name}': {p.shape}"
                )
            # flatten to (batch * steps,)
            flat_params[name] = p.reshape(batch_size * steps)

        return flat_params

    def sample(self, batch_size: int, steps: int):
        """
        Sample parameters from the prior and simulate data.
        """

        # Sample parameters
        prior_draws = self.prior.sample(batch_size=batch_size, steps=steps)
        local_params = prior_draws["local_params"]
        shared_params = prior_draws.get("shared_params")

        # Combine parameter dictionaries
        combined_params = dict(local_params)
        if shared_params is not None:
            combined_params.update(shared_params)

        # Broadcast + flatten params
        flat_params = self._prepare_flat_params(
            combined_params, batch_size, steps
        )

        # Order parameters according to model signature
        ordered_params = []
        for name in self.param_order:
            if name in flat_params:
                ordered_params.append(flat_params[name])
            else:
                # fallback to default value from function signature
                default = self.signature.parameters[name].default
                if default is inspect.Parameter.empty:
                    raise ValueError(
                        f"Parameter '{name}' required by model but missing in prior and has no default."
                    )
                ordered_params.append(default)

        # Run simulator
        sim_data = self.model(*ordered_params)
        sim_data = np.asarray(sim_data)

        # Reshape back to trajectories
        output_shape = sim_data.shape[1:] if sim_data.ndim > 1 else ()

        sim_data = sim_data.reshape(
            batch_size,
            steps,
            *output_shape
        )

        result = {
            "data": sim_data,
            "local_params": local_params,
            "global_params": prior_draws["global_params"],
        }

        if shared_params is not None:
            result["shared_params"] = shared_params

        return result