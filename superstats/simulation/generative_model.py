from typing import Callable, Dict, Optional
import inspect
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from superstats.prior.joint_prior import JointPrior
from superstats.diagnostics.plots.prior_push_forward import plot_push_forward


class GenerativeModel:
    """A generative model that combines a joint prior with a simulation function.

    This class facilitates sampling parameters from a joint prior distribution
    and generating simulated data using a user-provided model function. It handles
    parameter broadcasting, flattening, and reshaping to support batched simulations
    with time-varying parameters.

    Parameters
    ----------
    prior : JointPrior
        The joint prior distribution over model parameters, which may
        include both time-varying transitions and time-invariant priors.
    model : Callable
        The simulation function that takes parameter values and returns
        simulated data. The function signature determines the expected
        parameter names and order.

    Raises
    ------
    TypeError
        If `model` is not callable.
    """

    def __init__(
        self,
        prior: JointPrior,
        model: Callable
    ):
        self.prior = prior
        self.model = model

        if not callable(model):
            raise TypeError("model must be callable")

        # Inspect simulator signature
        self.signature = inspect.signature(model)
        self.param_order = list(self.signature.parameters.keys())

        # Run a pilot draw to determine key groups once
        pilot = self.prior.sample(batch_size=1, num_steps=1)
        self.local_keys = list(pilot["local_params"].keys()) if pilot.get("local_params") else []
        self.hyper_keys = list(pilot["hyper_params"].keys()) if pilot.get("hyper_params") else []
        self.shared_keys = list(pilot["shared_params"].keys()) if pilot.get("shared_params") else []
        self.fixed_keys = list(pilot["fixed_params"].keys()) if pilot.get("fixed_params") else []

    def _prepare_flat_params(
        self,
        combined_params: Dict[str, np.ndarray],
        batch_size: int,
        num_steps: int,
    ) -> Dict[str, np.ndarray]:
        """Broadcast and flatten parameters for vectorized simulation.

        Each entry in `combined_params` is broadcast to (batch_size,
        num_steps[, dim]) and flattened along the first two axes, so the
        simulator can be called once with 1D (or 2D, if `dim > 1`) inputs
        instead of being looped over trials and steps.

        Parameters
        ----------
        combined_params : dict of np.ndarray
            Mapping from model parameter name to a value of ndim 0, 1,
            2, or 3:
            - ndim 0 (scalar): broadcast to every trial and step.
            - ndim 1: shape (batch_size,), broadcast across steps.
            - ndim 2: shape (batch_size, num_steps), or
              (batch_size, dim) broadcast across steps.
            - ndim 3: shape (batch_size, num_steps, dim).
            Keys not present in `combined_params` are skipped if the
            model parameter has a default value.
        batch_size      : int
            Number of independent simulation batches.
        num_steps       : int
            Number of time steps per trajectory.

        Returns
        -------
        flat_params : dict of np.ndarray - mapping from parameter name
            to a flattened array of shape (batch_size * num_steps,) or
            (batch_size * num_steps, dim), ready to pass to `self.model`

        Raises
        ------
        ValueError
            If a required parameter (no default in the model signature)
            is missing from `combined_params`, or if a parameter's
            shape doesn't match any of the supported ndim-0/1/2/3 cases.
        """
        flat_params: Dict[str, np.ndarray] = {}

        for name in self.param_order:
            if name not in combined_params:
                param = self.signature.parameters[name]
                if param.default is inspect.Parameter.empty:
                    raise ValueError(
                        f"Parameter '{name}' required by model but missing in prior."
                    )
                continue

            p = np.asarray(combined_params[name])

            if p.ndim == 0:
                p = np.full((batch_size, num_steps), p.item(), dtype=p.dtype)
                flat_params[name] = p.reshape(batch_size * num_steps)
                continue

            if p.ndim == 1:
                if p.shape[0] != batch_size:
                    raise ValueError(
                        f"Parameter '{name}' must have shape (batch_size,) or (batch_size, num_steps); got {p.shape}"
                    )
                p = np.broadcast_to(p[:, None], (batch_size, num_steps))
                flat_params[name] = p.reshape(batch_size * num_steps)
                continue

            if p.ndim == 2:
                if p.shape == (batch_size, num_steps):
                    flat_params[name] = p.reshape(batch_size * num_steps)
                elif p.shape[0] == batch_size:
                    flat_params[name] = np.broadcast_to(
                        p[:, None, ...],
                        (batch_size, num_steps, p.shape[1])
                    ).reshape(batch_size * num_steps, p.shape[1])
                else:
                    raise ValueError(
                        f"Parameter '{name}' must have shape (batch_size, num_steps) or (batch_size, dim); got {p.shape}"
                    )
                continue

            if p.ndim == 3:
                if p.shape[0] != batch_size or p.shape[1] != num_steps:
                    raise ValueError(
                        f"Parameter '{name}' must have shape (batch_size, num_steps, dim); got {p.shape}"
                    )
                if p.shape[2] == 1:
                    flat_params[name] = p.reshape(batch_size * num_steps)
                else:
                    flat_params[name] = p.reshape(batch_size * num_steps, p.shape[2])
                continue

            raise ValueError(
                f"Unexpected shape for parameter '{name}': {p.shape}"
            )

        return flat_params

    def get_fixed_params(self) -> Dict[str, np.ndarray]:
        """Return deterministic fixed parameters from the prior for model simulation.

        Draws a single pilot sample from `self.prior` and keeps only the
        fixed-parameter entries that the model actually consumes.

        Returns
        -------
        fixed_params : dict of np.ndarray - mapping from parameter name
            to its fixed value, restricted to names in `self.param_order`
        """
        prior_draws = self.prior.sample(batch_size=1, num_steps=1)
        fixed_params = prior_draws.get("fixed_params", {})
        return {
            name: np.asarray(value)
            for name, value in fixed_params.items()
            if name in self.param_order
        }

    def simulate_from_parameters(
        self,
        params: Dict[str, np.ndarray],
        batch_size: int,
        num_steps: int,
    ) -> np.ndarray:
        """Simulate model outputs for given parameter values.

        Parameters
        ----------
        params     : dict of np.ndarray
            Parameter values to simulate from, keyed by model parameter
            name. See `_prepare_flat_params` for the accepted shapes.
        batch_size : int
            Number of independent simulation batches.
        num_steps  : int
            Number of time steps per trajectory.

        Returns
        -------
        sim_data : np.ndarray of shape (batch_size, num_steps, ...) -
            simulated data reshaped to trajectory format, where any
            trailing dimensions match `self.model`'s own output shape

        Raises
        ------
        ValueError
            If a required parameter is missing from `params` and has no
            default in the model signature, or has an unsupported shape.
        """
        combined_params = dict(params)

        flat_params = self._prepare_flat_params(
            combined_params,
            batch_size=batch_size,
            num_steps=num_steps,
        )

        ordered_params = []
        for name in self.param_order:
            if name in flat_params:
                ordered_params.append(flat_params[name])
                continue

            default = self.signature.parameters[name].default
            if default is inspect.Parameter.empty:
                raise ValueError(
                    f"Parameter '{name}' required by model but missing in params and has no default."
                )
            ordered_params.append(default)

        sim_data = self.model(*ordered_params)
        sim_data = np.asarray(sim_data)

        output_shape = sim_data.shape[1:] if sim_data.ndim > 1 else ()
        return sim_data.reshape(batch_size, num_steps, *output_shape)

    def _normalize_local_params(
        self,
        params: Dict[str, np.ndarray],
        batch_size: int,
        num_steps: int,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Validate and normalize local (time-varying) parameters.

        Parameters
        ----------
        params     : dict of np.ndarray
            Mapping from parameter name to an array of shape
            (batch_size, num_steps).
        batch_size : int
            Expected first-axis size for every parameter.
        num_steps  : int
            Expected second-axis size for every parameter.

        Returns
        -------
        normalized : dict of np.ndarray or None - each array reshaped
            to (batch_size, num_steps, 1); None if `params` is empty

        Raises
        ------
        ValueError
            If any parameter's shape is not exactly
            (batch_size, num_steps).
        """
        if not params:
            return None

        normalized: Dict[str, np.ndarray] = {}
        for name, value in params.items():
            arr = np.asarray(value)
            if arr.ndim != 2 or arr.shape != (batch_size, num_steps):
                raise ValueError(
                    f"Local parameter '{name}' must have shape (batch_size, num_steps), got {arr.shape}"
                )
            normalized[name] = arr.reshape(batch_size, num_steps, 1)
        return normalized

    def _normalize_batch_params(
        self,
        params: Dict[str, np.ndarray],
        batch_size: int,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Validate and normalize batch-level (non-time-varying) parameters.

        Parameters
        ----------
        params     : dict of np.ndarray
            Mapping from parameter name to a scalar, or an array of
            ndim 1 or 2.
        batch_size : int
            Expected first-axis size for 1D/2D parameters, and the
            number of copies to broadcast a scalar to.

        Returns
        -------
        normalized : dict of np.ndarray or None - each array reshaped
            or broadcast to (batch_size, dim); None if `params` is empty

        Raises
        ------
        ValueError
            If any parameter has ndim greater than 2.
        """
        if not params:
            return None

        normalized: Dict[str, np.ndarray] = {}

        for name, value in params.items():
            arr = np.asarray(value)

            if arr.ndim == 1:
                normalized[name] = arr.reshape(batch_size, 1)
            elif arr.ndim == 2 and arr.shape[1] == 1:
                normalized[name] = arr
            elif arr.ndim == 2:
                normalized[name] = arr
            elif arr.ndim == 0:
                normalized[name] = np.full((batch_size, 1), arr.item(), dtype=arr.dtype)
            else:
                raise ValueError(
                    f"Parameter '{name}' has invalid shape {arr.shape}"
                )

        return normalized

    def sample(
        self,
        batch_size: int,
        num_steps: int,
        include_fixed: bool = False,
        tile_to_steps: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Sample parameters from the prior and generate simulated data.

        This method performs a complete generative process:
        1. Samples parameters from the joint prior distribution
        2. Prepares parameters for vectorized simulation
        3. Runs the simulation model
        4. Reshapes outputs back to trajectory format

        Parameters
        ----------
        batch_size    : int
            Number of independent simulation batches to generate.
        num_steps     : int
            Number of time steps per trajectory.
        include_fixed : bool, optional, default: False
            If True, include `fixed_params` in the returned dictionary.
        tile_to_steps : bool, optional, default: False
            If True, tile `hyper_params` and `shared_params` from shape
            (batch_size, 1) to (batch_size, num_steps, 1), aligning
            them with the time axis of local parameters.

        Returns
        -------
        result : dict - flat dictionary with `'data'` plus one entry
            per sampled parameter.
            Local (time-varying) params have shape
            (batch_size, num_steps, 1); hyper and shared params have
            shape (batch_size, 1), or (batch_size, num_steps, 1) when
            `tile_to_steps` is True.
            Fixed params are included only when `include_fixed` is True.

            The instance attributes `local_keys`, `hyper_keys`,
            `shared_keys`, and `fixed_keys` are updated each call to
            record which keys belong to which parameter group.

        Raises
        ------
        ValueError
            If required parameters are missing from the prior or have
            invalid shapes.
        """
        # Sample parameters
        prior_draws = self.prior.sample(batch_size=batch_size, num_steps=num_steps)
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
            combined_params, batch_size, num_steps
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
            num_steps,
            *output_shape
        )

        local_params = self._normalize_local_params(local_params, batch_size, num_steps)
        hyper_params = self._normalize_batch_params(prior_draws.get("hyper_params", {}), batch_size)
        shared_params = self._normalize_batch_params(shared_params, batch_size)

        if tile_to_steps:
            if hyper_params is not None:
                hyper_params = {
                    k: np.tile(v[:, np.newaxis, :], (1, num_steps, 1))
                    for k, v in hyper_params.items()
                }
            if shared_params is not None:
                shared_params = {
                    k: np.tile(v[:, np.newaxis, :], (1, num_steps, 1))
                    for k, v in shared_params.items()
                }

        result = {"data": sim_data}
        if local_params:
            result.update(local_params)
        if hyper_params:
            result.update(hyper_params)
        if shared_params:
            result.update(shared_params)
        if include_fixed and fixed_params:
            result.update(fixed_params)

        return result

    def plot_push_forward(
        self,
        num_sim: int = 20,
        num_steps: int = 200,
        data_dim: int = 0,
        kind: str = "dist",
        aggregate_fun: str | Callable | None = None,
        uncertainty_fun: str | Callable | None = None,
        spaghetti: bool = True,
        marginal: bool = True,
        **kwargs,
    ) -> plt.Figure:
        """Render prior push-forward diagnostics for the generative model.

        Parameters
        ----------
        num_sim         : int, optional, default: 20
            Number of simulated datasets to generate.
        num_steps       : int, optional, default: 200
            Number of time steps per simulation.
        data_dim        : int, optional, default: 0
            Data dimension to plot.
        kind            : {"dist", "trajectory"}, optional, default: "dist"
            Plot type.
        aggregate_fun   : {"mean", "median"} or callable or None, optional, default: None
            Aggregation function over simulations.
        uncertainty_fun : {"std", "95ci", "mad", "95hdi"} or callable or None, optional, default: None
            Uncertainty function for aggregate trajectory plots. Forwarded
            directly to `plot_push_forward`, so the accepted values must
            match that function's own supported set.
        spaghetti       : bool, optional, default: True
            If True, include individual trajectories.
        marginal        : bool, optional, default: True
            If True, include marginal distributions beside trajectories.
        **kwargs
            Forwarded to `plot_push_forward`.

        Returns
        -------
        fig : plt.Figure - the figure containing the requested plot
        """
        data = self.sample(batch_size=num_sim, num_steps=num_steps)["data"]
        return plot_push_forward(
            data=data,
            data_dim=data_dim,
            kind=kind,
            aggregate_fun=aggregate_fun,
            uncertainty_fun=uncertainty_fun,
            spaghetti=spaghetti,
            marginal=marginal,
            **kwargs,
        )