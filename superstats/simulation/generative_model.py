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
        # Broadcast and flatten parameters for vectorized simulation.
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
            p = np.asarray(p)
            # shared parameters (batch,) → (batch, steps)
            if p.ndim == 1:
                p = np.broadcast_to(p[:, None], (batch_size, steps))
                flat_params[name] = p.reshape(batch_size * steps)
            # already trajectory parameters
            elif p.ndim == 2:
                flat_params[name] = p.reshape(batch_size * steps)
            # fixed scalar parameters also broadcast to full trajectory length
            elif p.ndim == 0:
                p = np.full((batch_size, steps), p.item(), dtype=np.asarray(p).dtype)
                flat_params[name] = p.reshape(batch_size * steps)
            else:
                raise ValueError(
                    f"Unexpected shape for parameter '{name}': {p.shape}"
                )

        return flat_params

    def _normalize_local_params(self, params, batch_size, steps):
        # Normalize time-varying parameters to shape (batch_size, steps, 1).
        if not params:
            return None

        normalized = {}
        for name, value in params.items():
            arr = np.asarray(value)
            if arr.ndim != 2:
                raise ValueError(
                    f"Local parameter '{name}' must have shape (batch_size, steps), got {arr.shape}"
                )
            normalized[name] = arr.reshape(batch_size, steps, 1)
        return normalized

    def _normalize_batch_params(self, params, batch_size):
        # Normalize scalar or batch parameters to shape (batch_size, 1).
        if not params:
            return None

        normalized = {}
        for name, value in params.items():
            arr = np.asarray(value)
            if arr.ndim == 1:
                normalized[name] = arr.reshape(batch_size, 1)
            elif arr.ndim == 2 and arr.shape[1] == 1:
                normalized[name] = arr
            else:
                raise ValueError(
                    f"Parameter '{name}' must have shape (batch_size,) or (batch_size, 1), got {arr.shape}"
                )
        return normalized

    def sample(self, batch_size: int, steps: int, include_fixed: bool = False, tile_to_steps: bool = False):
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
        include_fixed : bool, optional
            If True, include ``fixed_params`` in the returned dictionary.
            Default is False.
        tile_to_steps : bool, optional
            If True, tile ``hyper_params`` and ``shared_params`` from shape
            (batch_size, 1) to (batch_size, steps, 1), aligning them with
            the time axis of local parameters. Default is False.

        Returns
        -------
        dict
            Dictionary containing:
            - 'data': np.ndarray
                Simulated data of shape (batch_size, steps, ...) where additional
                dimensions depend on the model output.
            - 'local_params': dict
                Time-varying parameters for each trajectory, each shaped
                (batch_size, steps, 1).
            - 'hyper_params': dict or None
                Hyperparameters from transition processes, each shaped
                (batch_size, 1).
            - 'fixed_params': dict or None
                Fixed scalar parameters from the prior.
            - 'shared_params': dict or None
                Time-invariant parameters shared across trajectories, each shaped
                (batch_size, 1), or (batch_size, steps, 1) if ``tile_to_steps`` is True.
            - 'fixed_params': dict or None
                Fixed scalar parameters from the prior. Only included when
                ``include_fixed`` is True.

        Raises
        ------
        ValueError
            If required parameters are missing from the prior or have invalid shapes.
        """

        # Sample parameters
        prior_draws = self.prior.sample(batch_size=batch_size, steps=steps)
        local_params = prior_draws["local_params"]
        shared_params = prior_draws.get("shared_params", {})
        fixed_params = prior_draws.get("fixed_params", {})

        # Combine parameter dictionaries
        combined_params = dict(local_params)
        combined_params.update(shared_params)
        # Include fixed params that are used by the model
        for name in self.param_order:
            if name in fixed_params:
                combined_params[name] = fixed_params[name]

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

        local_params = self._normalize_local_params(local_params, batch_size, steps)
        hyper_params = self._normalize_batch_params(prior_draws.get("hyper_params", {}), batch_size)
        shared_params = self._normalize_batch_params(shared_params, batch_size)

        if tile_to_steps:
            if hyper_params is not None:
                hyper_params = {
                    k: np.tile(v[:, np.newaxis, :], (1, steps, 1))
                    for k, v in hyper_params.items()
                }
            if shared_params is not None:
                shared_params = {
                    k: np.tile(v[:, np.newaxis, :], (1, steps, 1))
                    for k, v in shared_params.items()
                }

        result = {
            "data": sim_data,
            "local_params": local_params,
            "hyper_params": hyper_params,
            "shared_params": shared_params,
        }

        if include_fixed:
            result["fixed_params"] = fixed_params if fixed_params else None

        return result
