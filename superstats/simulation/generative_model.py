from typing import Callable, Dict
import inspect
import numpy as np
from numba import njit, prange

from superstats.prior.joint_prior import JointPrior

# @njit(parallel=True)
# def sample_data_batch(model, stacked_params, sim_data):
#     batch_size, steps, num_params = stacked_params.shape

#     for b in prange(batch_size):
#         for t in range(steps):
#             params = stacked_params[b, t]
#             sim_data[b, t] = model(*params)

#     return sim_data

class GenerativeModel:
    def __init__(
        self,
        prior: JointPrior,
        model: Callable
    ):
        self.prior = prior
        self.model = model
        self.signature = inspect.signature(model)
        self.param_order = list(self.signature.parameters.keys())

    def sample(self, batch_size: int, steps: int):
        # Sample parameters
        prior_draws = self.prior.sample(batch_size=batch_size, steps=steps)
        local_params = prior_draws["local_params"]
        shared_params = prior_draws.get("shared_params")

        # Combine local and shared parameters
        combined_params = dict(local_params)
        if shared_params is not None:
            combined_params.update(shared_params)

        # Stack parameters according to model signature
        arrays = []
        for name, param in self.signature.parameters.items():
            if name not in combined_params:
                if param.default is inspect.Parameter.empty:
                    raise ValueError(f"Parameter {name} required by model but missing in prior.")
                else:
                    continue
            p = combined_params[name]
            if p.ndim == 1:
                arrays.append(np.broadcast_to(p[:, None], (batch_size, steps)))
            elif p.ndim == 2:
                arrays.append(p)
            else:
                raise ValueError(f"Unexpected shape for parameter {name}: {p.shape}")
            
        stacked_params = np.stack(arrays, axis=-1)

        # Infer output shape
        first = np.asarray(self.model(*stacked_params[0, 0]))
        sim_data = np.zeros((batch_size, steps) + first.shape, dtype=first.dtype)

        # simulate data
        for b in range(batch_size):
            for t in range(steps):
                params = stacked_params[b, t]
                sim_data[b, t] = self.model(*params)

        result = {
            "data": sim_data,
            "local_params": local_params,
            "global_params": prior_draws["global_params"],
        }

        if shared_params is not None:
            result["shared_params"] = shared_params

        return result
