from typing import Callable, Dict
import inspect
import numpy as np
from numba import njit, prange

from superstats.prior.joint_prior import JointPrior

class GenerativeModel:
    """
    A generative model that combines a joint prior with a simulation function.

    This class facilitates sampling parameters from a joint prior distribution
    and generating simulated data using a user-provided model function. It handles
    parameter broadcasting, flattening, and reshaping to support batched simulations
    with time-varying parameters.

    Parameters
    ----------
    prior : JointPrior
        The joint prior distribution over model parameters, which may include
        both time-varying transitions and time-invariant priors.
    model : Callable
        The simulation function that takes parameter values and returns simulated data.
        The function signature determines the expected parameter names and order.
    """

    def __init__(
        self,
        prior: JointPrior,
        model: Callable
    ):
        """
        Initialize the generative model.

        Parameters
        ----------
        prior : JointPrior
            The joint prior distribution for model parameters.
        model : Callable
            The simulation function to generate data from parameters.
        """
        self.prior = prior
        self.model = model
        # Inspect simulator signature
        self.signature = inspect.signature(model)
        self.param_order = list(self.signature.parameters.keys())

    def _prepare_flat_params(self, combined_params, batch_size, steps):
        """
        Prepare parameters for simulation by broadcasting and flattening.

        This method takes combined parameter dictionaries and prepares them for
        the simulation function by:
        1. Broadcasting shared parameters from (batch,) to (batch, steps)
        2. Flattening all parameters to (batch*steps,) for vectorized simulation
        3. Ordering parameters according to the model function signature

        Parameters
        ----------
        combined_params : dict
            Dictionary mapping parameter names to arrays of shape (batch,) or (batch, steps).
        batch_size : int
            Number of independent simulation batches.
        steps : int
            Number of time steps per trajectory.

        Returns
        -------
        dict
            Dictionary mapping parameter names to flattened arrays of shape (batch*steps,).

        Raises
        ------
        ValueError
            If a required parameter is missing from combined_params and has no default value,
            or if parameter shapes are unexpected.
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
        Sample parameters from the prior and generate simulated data.

        This method performs a complete generative process:
        1. Samples parameters from the joint prior distribution
        2. Prepares parameters for vectorized simulation
        3. Runs the simulation model
        4. Reshapes outputs back to trajectory format

        Parameters
        ----------
        batch_size : int
            Number of independent simulation batches to generate.
        steps : int
            Number of time steps per trajectory.

        Returns
        -------
        dict
            Dictionary containing:
            - 'data': np.ndarray
                Simulated data of shape (batch_size, steps, ...) where additional
                dimensions depend on the model output.
            - 'local_params': dict
                Time-varying parameters for each trajectory.
            - 'global_params': dict
                Hyperparameters from transition processes.
            - 'shared_params': dict, optional
                Time-invariant parameters shared across trajectories.

        Raises
        ------
        ValueError
            If required parameters are missing from the prior or have invalid shapes.
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

        if "infer_mask" in prior_draws:
            result["infer_mask"] = prior_draws["infer_mask"]

        return result