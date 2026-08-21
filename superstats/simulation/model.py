"""Generative-simulator wrapper for joint priors and simulators."""

from typing import Callable, Dict, Optional, Literal
from collections.abc import Mapping, Sequence
import inspect
import numpy as np
import matplotlib.pyplot as plt

from superstats.prior.joint_prior import JointPrior
from superstats.diagnostics.plots.prior_push_forward import plot_push_forward
from superstats.simulation.augmentation.missing import MissingProcess
from superstats.simulation.augmentation.contamination import ContaminationProcess
from superstats.utils.dispatch import find_contamination, find_missing


class Model:
    """A generative simulator that combines a joint prior with a simulation function.

    This class facilitates sampling parameters from a joint prior distribution
    and generating simulated data using a user-provided simulator function. It handles
    parameter broadcasting, flattening, and reshaping to support batched
    simulations with time-varying parameters. Optionally, a missing-data process
    can be applied to the simulated data to introduce and record missingness.

    Parameters
    ----------
    prior : JointPrior
        The joint prior distribution over simulator parameters, which may
        include both time-varying transitions and time-invariant priors.
    simulator : Callable
        The simulation function that takes parameter values and returns
        simulated data. The function signature determines the expected
        parameter names and order.
    missing : MissingProcess, Callable, "random", or None, optional, default: "random"
        Process applied to simulated data to introduce missingness.
        - Not provided (default) or `"random"`: uses `RandomMissingProcess()`,
          the default MCAR missingness process.
        - `None`: disables missingness augmentation and `sample` will not
          include a `"missing_mask"` entry in its result.
        - `MissingProcess` instance: used as-is.
        - Plain `Callable`: must follow the same contract as
          `MissingProcess.__call__`, i.e.
          `(data_mapping, rng=None) -> filled_mapping | {"missing_mask": mask}`.
    contamination : ContaminationProcess, Callable, "random_choice", or None, optional, default: None
        Process applied to simulated observations before missingness. A
        `RandomChoiceContamination` configured with `infer=True` contributes
        its probability and transition parameters to this model's parameter
        categories.

    Raises
    ------
    TypeError
        If `simulator` is not callable, or if `missing` is neither
        `None`, `"random"`, nor callable.
    """

    def __init__(
        self,
        prior: JointPrior,
        simulator: Callable,
        missing: MissingProcess | Callable | Literal["random"] | None = "random",
        contamination: ContaminationProcess | Callable | Literal["random_choice"] | None = None,
    ):
        self.prior = prior
        self.simulator = simulator

        self.missing = find_missing(missing)

        if self.missing is not None:
            self.has_mask = True
        else:
            self.has_mask = False

        self.contamination = find_contamination(contamination)

        # Inspect simulator signature
        self.signature = inspect.signature(simulator)
        self.param_order = list(self.signature.parameters.keys())

        # Run a pilot draw to determine key groups once
        pilot = self.prior.sample(batch_size=1, num_steps=1)
        self.local_keys = list(pilot["local_params"].keys()) if pilot.get("local_params") else []
        self.deterministic_keys = (
            list(pilot["deterministic_params"].keys()) if pilot.get("deterministic_params") else []
        )
        self.hyper_keys = list(pilot["hyper_params"].keys()) if pilot.get("hyper_params") else []
        self.shared_keys = list(pilot["shared_params"].keys()) if pilot.get("shared_params") else []
        self.fixed_keys = list(pilot["fixed_params"].keys()) if pilot.get("fixed_params") else []

        self._contamination_parameter_groups = {}
        if self.contamination is not None and hasattr(self.contamination, "parameter_groups"):
            self._contamination_parameter_groups = self.contamination.parameter_groups()
            model_groups = {
                "local_params": self.local_keys,
                "deterministic_params": self.deterministic_keys,
                "hyper_params": self.hyper_keys,
                "shared_params": self.shared_keys,
                "fixed_params": self.fixed_keys,
            }
            existing_keys = set().union(*model_groups.values())
            contamination_keys = {key for keys in self._contamination_parameter_groups.values() for key in keys}
            overlap = existing_keys & contamination_keys
            if overlap:
                raise ValueError(f"Contamination parameter names conflict with prior parameters: {sorted(overlap)}")
            for group, keys in self._contamination_parameter_groups.items():
                model_groups[group].extend(keys)

        self.data_keys = self._infer_data_keys(pilot)

    def _ordered_model_args(
        self,
        combined_params: Dict[str, np.ndarray],
        batch_size: int,
        num_steps: int,
        missing_context: str,
    ) -> list:
        """Prepare simulator arguments in signature order."""
        flat_params = self._prepare_flat_params(
            combined_params,
            batch_size=batch_size,
            num_steps=num_steps,
            missing_context=missing_context,
        )

        ordered_params = []
        for name in self.param_order:
            if name in flat_params:
                ordered_params.append(flat_params[name])
                continue

            default = self.signature.parameters[name].default
            if default is inspect.Parameter.empty:
                raise ValueError(f"Parameter '{name}' required by simulator but missing in {missing_context}.")
            ordered_params.append(default)

        return ordered_params

    def _reshape_model_output(
        self,
        model_output: Mapping[str, np.ndarray],
        batch_size: int,
        num_steps: int,
        expected_data_keys: Sequence[str] | None = None,
    ) -> Dict[str, np.ndarray]:
        """Validate and reshape a named simulator output dict."""
        if not isinstance(model_output, Mapping):
            raise TypeError(f"simulator must return a dict of named arrays, got {type(model_output)}.")
        if not model_output:
            raise ValueError("simulator must return a non-empty dict of named arrays.")

        expected_shape = (batch_size * num_steps,)
        reshaped = {}

        for name, value in model_output.items():
            if not isinstance(name, str):
                raise TypeError(f"simulator output keys must be strings, got {name!r}.")

            arr = np.asarray(value)
            if arr.shape != expected_shape:
                raise ValueError(
                    f"Model output '{name}' must have shape {expected_shape} before reshaping, got {arr.shape}."
                )
            reshaped[name] = arr.reshape(batch_size, num_steps)

        data_keys = list(reshaped.keys())
        if expected_data_keys is not None and data_keys != list(expected_data_keys):
            raise ValueError(f"Model output keys changed from {list(expected_data_keys)!r} to {data_keys!r}.")

        return reshaped

    def _infer_data_keys(self, prior_draws: dict) -> list[str]:
        """Infer observation names from a one-step simulator call."""
        combined_params = dict(prior_draws.get("local_params", {}))
        combined_params.update(prior_draws.get("deterministic_params", {}))
        combined_params.update(prior_draws.get("shared_params", {}))

        fixed_params = prior_draws.get("fixed_params", {})
        for name in self.param_order:
            if name in fixed_params:
                combined_params[name] = fixed_params[name]

        ordered_params = self._ordered_model_args(
            combined_params,
            batch_size=1,
            num_steps=1,
            missing_context="prior",
        )
        model_output = self.simulator(*ordered_params)
        return list(self._reshape_model_output(model_output, batch_size=1, num_steps=1).keys())

    def _prepare_flat_params(
        self,
        combined_params: Dict[str, np.ndarray],
        batch_size: int,
        num_steps: int,
        missing_context: str = "prior",
    ) -> Dict[str, np.ndarray]:
        """Broadcast and flatten parameters for vectorized simulation.

        Each entry in `combined_params` is broadcast to (batch_size,
        num_steps[, dim]) and flattened along the first two axes, so the
        simulator can be called once with 1D (or 2D, if `dim > 1`) inputs
        instead of being looped over trials and steps.

        Parameters
        ----------
        combined_params : dict of np.ndarray
            Mapping from simulator parameter name to a value of ndim 0, 1,
            2, or 3:
            - ndim 0 (scalar): broadcast to every trial and step.
            - ndim 1: shape (batch_size,), broadcast across steps.
            - ndim 2: shape (batch_size, num_steps), or
              (batch_size, dim) broadcast across steps.
            - ndim 3: shape (batch_size, num_steps, dim).
            Keys not present in `combined_params` are skipped if the
            simulator parameter has a default value.
        batch_size      : int
            Number of independent simulation batches.
        num_steps       : int
            Number of time steps per trajectory.

        Returns
        -------
        flat_params : dict of np.ndarray - mapping from parameter name
            to a flattened array of shape (batch_size * num_steps,) or
            (batch_size * num_steps, dim), ready to pass to `self.simulator`

        Raises
        ------
        ValueError
            If a required parameter (no default in the simulator signature)
            is missing from `combined_params`, or if a parameter's
            shape doesn't match any of the supported ndim-0/1/2/3 cases.
        """
        flat_params = {}

        for name in self.param_order:
            if name not in combined_params:
                param = self.signature.parameters[name]
                if param.default is inspect.Parameter.empty:
                    raise ValueError(f"Parameter '{name}' required by simulator but missing in {missing_context}.")
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
                    flat_params[name] = np.broadcast_to(p[:, None, ...], (batch_size, num_steps, p.shape[1])).reshape(
                        batch_size * num_steps, p.shape[1]
                    )
                else:
                    raise ValueError(
                        f"Parameter '{name}' must have shape (batch_size, num_steps) or "
                        f"(batch_size, dim); got {p.shape}"
                    )
                continue

            if p.ndim == 3:
                if p.shape[0] != batch_size or p.shape[1] != num_steps:
                    raise ValueError(f"Parameter '{name}' must have shape (batch_size, num_steps, dim); got {p.shape}")
                if p.shape[2] == 1:
                    flat_params[name] = p.reshape(batch_size * num_steps)
                else:
                    flat_params[name] = p.reshape(batch_size * num_steps, p.shape[2])
                continue

            raise ValueError(f"Unexpected shape for parameter '{name}': {p.shape}")

        return flat_params

    def get_fixed_params(self) -> Dict[str, np.ndarray]:
        """Return deterministic fixed parameters from the prior for simulator simulation.

        Draws a single pilot sample from `self.prior` and keeps only the
        fixed-parameter entries that the simulator actually consumes.

        Returns
        -------
        fixed_params : dict of np.ndarray - mapping from parameter name
            to its fixed value, restricted to names in `self.param_order`
        """
        prior_draws = self.prior.sample(batch_size=1, num_steps=1)
        fixed_params = prior_draws.get("fixed_params", {})
        return {name: np.asarray(value) for name, value in fixed_params.items() if name in self.param_order}

    def simulate_from_parameters(
        self,
        params: Dict[str, np.ndarray],
        batch_size: int,
        num_steps: int,
    ) -> Dict[str, np.ndarray]:
        """Simulate simulator outputs for given parameter values.

        Parameters
        ----------
        params     : dict of np.ndarray
            Parameter values to simulate from, keyed by simulator parameter
            name. See `_prepare_flat_params` for the accepted shapes.
        batch_size : int
            Number of independent simulation batches.
        num_steps  : int
            Number of time steps per trajectory.

        Returns
        -------
        sim_data : dict of np.ndarray
            Named simulated variables. Each value has shape
            (batch_size, num_steps).

        Raises
        ------
        ValueError
            If a required parameter is missing from `params` and has no
            default in the simulator signature, or has an unsupported shape.
        """
        combined_params = dict(params)

        ordered_params = self._ordered_model_args(
            combined_params,
            batch_size=batch_size,
            num_steps=num_steps,
            missing_context="params and has no default",
        )

        model_output = self.simulator(*ordered_params)
        return self._reshape_model_output(model_output, batch_size, num_steps, expected_data_keys=self.data_keys)

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

        normalized = {}
        for name, value in params.items():
            arr = np.asarray(value)
            if arr.ndim != 2 or arr.shape != (batch_size, num_steps):
                raise ValueError(f"Local parameter '{name}' must have shape (batch_size, num_steps), got {arr.shape}")
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

        normalized = {}

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
                raise ValueError(f"Parameter '{name}' has invalid shape {arr.shape}")

        return normalized

    def _apply_contamination(
        self,
        sim_data: Dict[str, np.ndarray],
        rng: np.random.Generator | None,
    ) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Run `self.contamination` on `sim_data`, if configured.

        Parameters
        ----------
        sim_data : dict of np.ndarray
            Named simulated variables. Must include "response_time" and
            "choice" (each shape (batch_size, num_steps)) if a contamination
            process is configured, since `ContaminationProcess.apply`
            requires both. Any additional keys are passed through unchanged.
        rng      : np.random.Generator or None
            Generator forwarded to the contamination process.

        Returns
        -------
        sim_data : dict of np.ndarray - the (possibly contaminated) named
            simulated variables. "response_time" and "choice" are replaced
            by their contaminated versions if a process is configured;
            otherwise `sim_data` is returned unchanged.
        extra    : dict of np.ndarray - additional entries the process
            returned beyond the original `sim_data` keys (e.g.
            `"p_contaminated"` for `RandomChoiceContamination`); empty dict if
            `self.contamination` is None or the process returned no
            extra keys.
        """
        if self.contamination is None:
            return sim_data, {}

        if isinstance(self.contamination, ContaminationProcess):
            out = self.contamination.apply(sim_data, rng=rng)
        else:
            out = self.contamination(sim_data, rng=rng)

        extra_keys = out.keys() - sim_data.keys()
        extra = {key: out[key] for key in extra_keys}

        sim_data = {key: out[key] for key in sim_data.keys()}

        return sim_data, extra

    def _apply_missing(
        self,
        sim_data: Dict[str, np.ndarray],
        rng: np.random.Generator | None,
    ) -> tuple[Dict[str, np.ndarray], Optional[np.ndarray], Dict[str, np.ndarray]]:
        """Run `self.missing` on `sim_data`, if configured.

        Parameters
        ----------
        sim_data : dict of np.ndarray
            Named simulated variables to potentially corrupt with
            missingness. Each value must have shape
            (batch_size, num_steps).
        rng      : np.random.Generator or None
            Generator forwarded to the missing process, if it accepts one.

        Returns
        -------
        sim_data     : dict of np.ndarray - the (possibly corrupted)
            named simulated variables
        missing_mask : np.ndarray or None - mask from the process, or
            None if `self.missing` is None
        extra        : dict of np.ndarray - any additional entries the
            process returned beyond the simulator data keys and
            `"missing_mask"` (e.g. `RandomMissingProcess` also returns
            `"p_missing"`); empty dict if `self.missing` is None
            or the process returned no extra keys
        """
        if self.missing is None:
            return sim_data, None, {}

        try:
            params = inspect.signature(self.missing).parameters
            accepts_rng = "rng" in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
        except (TypeError, ValueError):
            accepts_rng = False

        result = self.missing(sim_data, rng=rng) if accepts_rng else self.missing(sim_data)

        sim_data = {key: result[key] for key in self.data_keys}
        missing_mask = result["missing_mask"]
        extra = {k: v for k, v in result.items() if k not in (*self.data_keys, "missing_mask")}
        return sim_data, missing_mask, extra

    def sample(
        self,
        batch_size: int,
        num_steps: int,
        include_fixed: bool = False,
        tile_to_steps: bool = False,
        rng: np.random.Generator | None = None,
    ) -> Dict[str, np.ndarray]:
        """Sample parameters from the prior and generate simulated data.

        This method performs a complete generative process:
        1. Samples parameters from the joint prior distribution
        2. Prepares parameters for vectorized simulation
        3. Runs the simulation simulator
        4. Reshapes outputs back to trajectory format
        5. Applies `self.contamination` and `self.missing`, if configured

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
        rng           : np.random.Generator or None, optional, default: None
            Random generator forwarded to `self.missing`. If
            None, the missing process falls back to its own default
            (an unseeded generator).

        Returns
        -------
        result : dict - flat dictionary with the following entries:
            - one entry per simulated observation variable, each with
            shape (batch_size, num_steps), corrupted by
            `self.missing` if one is configured.
            - `"time_steps"`: shape (batch_size, num_steps), each row
            equal to `1..num_steps`.
            - `"missing_mask"`: included only if `self.missing`
            is not None; shape matches the mask returned by the process
            (for `RandomMissingProcess`, (batch_size, num_steps)).
            - any additional keys the missing process returns beyond the
            simulator data keys and `"missing_mask"` (e.g.
            `RandomMissingProcess` also returns `"p_missing"`, shape
            (batch_size, 1)); omitted if `self.missing` is None
            or returns no extra keys.
            - contamination probabilities and transition parameters. When
              the contamination process has `infer=True`, these are shaped
              and returned with their registered model-parameter category;
              otherwise they remain augmentation metadata.
            - one entry per sampled parameter. Local (time-varying) params
              have shape (batch_size, num_steps); hyper and shared
              params have shape (batch_size, 1), or (batch_size, num_steps, 1)
              when `tile_to_steps` is True.
            - fixed params are included only when `include_fixed` is True.

            The instance attributes `local_keys`, `hyper_keys`,
            `shared_keys`, `fixed_keys`, and `data_keys` record which
            keys belong to which group.

        Raises
        ------
        ValueError
            If required parameters are missing from the prior or have
            invalid shapes.
        """
        # Sample parameters
        prior_draws = self.prior.sample(batch_size=batch_size, num_steps=num_steps)
        local_params = prior_draws["local_params"]
        deterministic_params = prior_draws.get("deterministic_params", {})
        shared_params = prior_draws.get("shared_params", {})
        fixed_params = prior_draws.get("fixed_params", {})

        # Combine parameter dictionaries
        combined_params = dict(local_params)
        combined_params.update(deterministic_params)
        combined_params.update(shared_params)

        # Include fixed params that are used by the simulator
        for name in self.param_order:
            if name in fixed_params:
                combined_params[name] = fixed_params[name]

        ordered_params = self._ordered_model_args(
            combined_params,
            batch_size=batch_size,
            num_steps=num_steps,
            missing_context="prior and has no default",
        )

        # Run simulator
        model_output = self.simulator(*ordered_params)
        sim_data = self._reshape_model_output(model_output, batch_size, num_steps, expected_data_keys=self.data_keys)

        # Apply contamination augmentation, if configured
        sim_data, contamination_extra = self._apply_contamination(sim_data, rng)

        for group, keys in self._contamination_parameter_groups.items():
            destination = {
                "local_params": local_params,
                "deterministic_params": deterministic_params,
                "hyper_params": prior_draws["hyper_params"],
                "shared_params": shared_params,
                "fixed_params": fixed_params,
            }[group]
            for key in keys:
                destination[key] = contamination_extra.pop(key)

        # Apply missingness augmentation, if configured
        sim_data, missing_mask, missing_extra = self._apply_missing(sim_data, rng)

        local_params = self._normalize_local_params(local_params, batch_size, num_steps)
        deterministic_params = self._normalize_local_params(deterministic_params, batch_size, num_steps)
        hyper_params = self._normalize_batch_params(prior_draws.get("hyper_params", {}), batch_size)
        shared_params = self._normalize_batch_params(shared_params, batch_size)

        if tile_to_steps:
            if hyper_params is not None:
                hyper_params = {k: np.tile(v[:, np.newaxis, :], (1, num_steps, 1)) for k, v in hyper_params.items()}
            if shared_params is not None:
                shared_params = {k: np.tile(v[:, np.newaxis, :], (1, num_steps, 1)) for k, v in shared_params.items()}

        time_steps = np.broadcast_to(np.arange(1, num_steps + 1)[None, :], (batch_size, num_steps))

        result = {**sim_data, "time_steps": time_steps}
        if contamination_extra:
            result.update(contamination_extra)
        if missing_mask is not None:
            result["missing_mask"] = missing_mask
        if missing_extra:
            result.update(missing_extra)
        if local_params:
            result.update(local_params)
        if deterministic_params:
            result.update(deterministic_params)
        if hyper_params:
            result.update(hyper_params)
        if shared_params:
            result.update(shared_params)
        if include_fixed and fixed_params:
            result.update(fixed_params)

        return result

    def plot_push_forward(
        self,
        batch_size: int = 20,
        num_steps: int = 200,
        data_dim: int | str = 0,
        kind: Literal["time_series", "dist"] = "dist",
        aggregation: Callable | None = None,
        uncertainty_fun: str | Callable | None = None,
        marginal: bool = True,
        dist_type: Literal["hist", "kde", "both"] = "hist",
        num_bins: int | None = None,
        dist_alpha: float | None = None,
        spaghetti: bool = False,
        num_cols: int | None = None,
        **kwargs,
    ) -> plt.Figure:
        """Render prior push-forward diagnostics for the generative simulator.

        Parameters
        ----------
        batch_size      : int, optional, default: 20
            Number of simulated datasets to generate.
        num_steps       : int, optional, default: 200
            Number of time steps per simulation.
        data_dim        : int or str, optional, default: 0
            Observation variable to plot. Integers index
            `self.data_keys`; strings select a variable by name.
        kind            : {"dist", "time_series"}, optional, default: "dist"
            Plot type.
        aggregation     : callable or None, optional, default: None
            Aggregation function over the dataset dimension, called as
            `aggregation(x, axis=...)` (e.g. np.mean, np.median).
            If None, individual datasets are shown in separate panels.
            If specified, all datasets are aggregated into a single panel.
        uncertainty_fun : {"std", "95ci", "mad", "95hdi"} or callable or None, optional, default: None
            Uncertainty function for aggregate time-series plots. Forwarded
            directly to `plot_push_forward`, so the accepted values must
            match that function's own supported set.
        marginal        : bool, optional, default: True
            If True, include marginal distributions beside time-series plots.
        dist_type       : {"hist", "kde", "both"}, optional, default: "hist"
            Distribution type used for continuous distributions and marginals.
        num_bins        : int or None, optional, default: None
            Number of histogram bins. If None, Seaborn selects the bins.
        dist_alpha      : float or None, optional, default: None
            Opacity of distributions and marginal distributions. If None,
            uses 1.0 for one distribution and 0.5 for overlays.
        spaghetti       : bool, optional, default: False
            If True, include individual time series.
        num_cols        : int or None, optional, default: None
            Number of panel columns. If None, uses the compact dynamic layout.
        **kwargs
            Forwarded to `plot_push_forward`.

        Returns
        -------
        fig : plt.Figure - the figure containing the requested plot
        """
        sample = self.sample(batch_size=batch_size, num_steps=num_steps)
        data = {key: sample[key] for key in self.data_keys}
        return plot_push_forward(
            data=data,
            data_dim=data_dim,
            kind=kind,
            aggregation=aggregation,
            uncertainty_fun=uncertainty_fun,
            spaghetti=spaghetti,
            marginal=marginal,
            dist_type=dist_type,
            num_bins=num_bins,
            dist_alpha=dist_alpha,
            num_cols=num_cols,
            **kwargs,
        )
