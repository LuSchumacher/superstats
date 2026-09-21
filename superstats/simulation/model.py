"""Generative-simulator wrapper for joint priors and simulators."""

from typing import Any, Callable, Dict, Optional, Literal
from collections.abc import Mapping, Sequence
import inspect
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from superstats.prior.joint_prior import JointPrior
from .link_function import LinkFunction
from matplotlib.figure import Figure
from superstats.defaults import BASE_COLOR, LABEL_FONTSIZE, TICK_FONTSIZE, TITLE_FONTSIZE
from superstats.diagnostics.plots.joint_prior import plot_joint_prior
from superstats.diagnostics.plots.time_invariant_prior import plot_time_invariant_prior
from superstats.diagnostics.plots.time_varying_prior import plot_time_varying_prior
from .augmentation.random_choice_contamination import RandomChoiceContamination
from superstats.diagnostics.plots.prior_push_forward import plot_push_forward
from superstats.simulation.augmentation.random_missing import RandomMissingProcess
from superstats.simulation.augmentation.missing import MissingProcess
from superstats.simulation.augmentation.contamination import ContaminationProcess
from superstats.simulation.context.context_simulator import ContextSimulator
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
    prior             : JointPrior
        The joint prior distribution over simulator parameters, which may
        include both time-varying transitions and time-invariant priors.
    simulator         : Callable
        The simulation function that takes parameter values and returns
        simulated data. The function signature determines the expected
        parameter names and order.
    link_function     : mapping, LinkFunction, callable, or None, optional
        Output links applied after regression and parameter context binding.
        A mapping assigns a link per simulator parameter; omitted entries use
        identity. A single link applies to all supplied simulator parameters.
        Inference targets always remain on the raw coefficient scale.
    formula           : callable or object with ``resolve``, optional, default: None
        Parameter resolver called as ``resolve(parameters=..., context=...)``
        before the simulator runs.
    context           : ContextSimulator, callable, Mapping, pandas.DataFrame, or None, optional, default: None
        Source of externally defined context variables. A batched callable
        accepting batch_size and num_steps is wrapped as a ContextSimulator.
        A ``ContextSimulator``
        generates new context for every sample. A mapping or DataFrame is
        treated as fixed trial-level context and repeated across the batch;
        DataFrame columns become context variables.
    simulator_context : sequence of str, optional, default: ()
        Context variable names bound to matching simulator parameters or
        forwarded through its context keyword argument.
    design_context    : sequence of str, optional, default: ()
        Context variable names routed to Formula. Variables can also be
        included in simulator_context.
    missing           : MissingProcess, Callable, "random", or None, optional, default: "random"
        Process applied to simulated data to introduce missingness.
        - Not provided (default) or `"random"`: uses `RandomMissingProcess()`,
          the default MCAR missingness process.
        The nuisance probability `p_missing` belongs in `JointPrior`; Model
        adds the default beta prior when omitted, without mutating the caller's
        prior. It and its transition hyperparameters are never inferred.
        - `None`: disables missingness augmentation and `sample` will not
          include a `"missing_mask"` entry in its result.
        - `MissingProcess` instance: used as-is.
        - Plain `Callable`: must follow the same contract as
          `MissingProcess.__call__`, i.e.
          `(data_mapping, rng=None) -> filled_mapping | {"missing_mask": mask}`.
    contamination     : ContaminationProcess, Callable, "random_choice", or None, optional, default: None
        Process applied to simulated observations before missingness. A
        `RandomChoiceContamination` configured with `infer=True` contributes
        the probability specified as `p_contaminated` in `JointPrior` and
        its sampled transition hyperparameters to inference targets. The
        process defaults to `infer=False`. If omitted from the prior, the
        default beta probability prior is added without mutating the caller's
        prior. Nuisance probabilities are redrawn during resimulation.

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
        link_function: Mapping[str, LinkFunction | Callable] | LinkFunction | Callable | None = None,
        formula: Any | None = None,
        context: ContextSimulator | Callable | Mapping[str, Any] | pd.DataFrame | None = None,
        simulator_context: Sequence[str] = (),
        design_context: Sequence[str] = (),
        missing: MissingProcess | Callable | Literal["random"] | None = "random",
        contamination: ContaminationProcess | Callable | Literal["random_choice"] | None = None,
    ):
        self.prior = prior
        self.simulator = simulator
        if callable(context) and not isinstance(context, ContextSimulator):
            context = ContextSimulator(context)
        self.context = context
        self.design_context = self._context_names(design_context, "design_context")
        self.simulator_context = self._context_names(simulator_context, "simulator_context")
        self.formula = formula
        if link_function is None:
            self.link_function = {}
        elif isinstance(link_function, Mapping):
            self.link_function = {
                name: link if isinstance(link, LinkFunction) else LinkFunction(link)
                for name, link in link_function.items()
            }
        elif callable(link_function):
            self.link_function = (
                link_function if isinstance(link_function, LinkFunction) else LinkFunction(link_function)
            )
        else:
            raise TypeError("link_function must be a mapping, LinkFunction, callable, or None.")
        if (self.design_context or self.simulator_context) and context is None:
            raise ValueError("Context selectors require context.")
        if context is not None and not isinstance(context, (ContextSimulator, Mapping, pd.DataFrame)):
            raise TypeError("context must be a ContextSimulator, callable, mapping, pandas DataFrame, or None.")

        self.missing = find_missing(missing)

        if self.missing is not None:
            self.has_mask = True
        else:
            self.has_mask = False

        self.contamination = find_contamination(contamination)
        if isinstance(self.contamination, RandomChoiceContamination) and "p_contaminated" not in self.prior.params:
            from superstats.defaults import DEFAULT_P_CONTAMINATED_PRIOR

            # Preserve the caller's prior while retaining the convenient default.
            self.prior = JointPrior(**self.prior.params, p_contaminated=DEFAULT_P_CONTAMINATED_PRIOR)

        if isinstance(self.missing, RandomMissingProcess) and "p_missing" not in self.prior.params:
            from superstats.defaults import DEFAULT_P_MISSING_PRIOR

            self.prior = JointPrior(**self.prior.params, p_missing=DEFAULT_P_MISSING_PRIOR)

        # Inspect simulator signature
        self.signature = inspect.signature(simulator)
        self.param_order = [name for name in self.signature.parameters if name != "context"]

        if isinstance(self.link_function, Mapping):
            unknown = set(self.link_function) - set(self.param_order) - {"p_contaminated", "p_missing"}
            if unknown:
                raise ValueError(f"Unknown link_function targets: {sorted(unknown)}")
            if "p_contaminated" in self.link_function and not isinstance(self.contamination, RandomChoiceContamination):
                raise ValueError("p_contaminated link requires RandomChoiceContamination.")
            if "p_missing" in self.link_function and not isinstance(self.missing, RandomMissingProcess):
                raise ValueError("p_missing link requires RandomMissingProcess.")

        # Run a pilot draw to determine key groups once
        pilot_context, pilot_contexts = self._generate_context(batch_size=1, num_steps=1, pilot=True)
        pilot = self.prior.sample(batch_size=1, num_steps=1)
        self.local_keys = list(pilot["local_params"].keys()) if pilot.get("local_params") else []
        self.deterministic_keys = (
            list(pilot["deterministic_params"].keys()) if pilot.get("deterministic_params") else []
        )
        self.hyper_keys = list(pilot["hyper_params"].keys()) if pilot.get("hyper_params") else []
        self.shared_keys = list(pilot["shared_params"].keys()) if pilot.get("shared_params") else []
        self.fixed_keys = list(pilot["fixed_params"].keys()) if pilot.get("fixed_params") else []

        self._contamination_parameter_groups = {}
        self._nuisance_keys = set()
        if isinstance(self.contamination, RandomChoiceContamination):
            owned_keys = {"p_contaminated"}
            owned_keys.update(self.prior._last_hyper_param_groups.get("p_contaminated", ()))
            owned_keys.update(self.prior._last_fixed_param_groups.get("p_contaminated", ()))
            self._contamination_parameter_groups = {
                group: [key for key in values if key in owned_keys] for group, values in pilot.items()
            }
            if not self.contamination.infer:
                self._nuisance_keys = {key for keys in self._contamination_parameter_groups.values() for key in keys}
        if "p_missing" in self.prior.params:
            self._nuisance_keys.add("p_missing")
            self._nuisance_keys.update(self.prior._last_hyper_param_groups.get("p_missing", ()))
            self._nuisance_keys.update(self.prior._last_fixed_param_groups.get("p_missing", ()))
        for attribute in ("local_keys", "deterministic_keys", "hyper_keys", "shared_keys", "fixed_keys"):
            setattr(self, attribute, [key for key in getattr(self, attribute) if key not in self._nuisance_keys])

        self.data_keys = self._infer_data_keys(pilot, pilot_contexts)
        self.context_keys = list(pilot_context)
        # Derived fields never replace raw parameters, observations, or generated context.
        protected_keys = set(self.data_keys) | set(self.context_keys) | {"time_steps", "missing_mask"}
        self.formula_keys = [name for name in self.formula_keys if name not in protected_keys]
        overlapping_keys = set(self.data_keys) & set(self.context_keys)
        if overlapping_keys:
            raise ValueError(f"Generated context keys conflict with simulator output keys: {sorted(overlapping_keys)}")
        self.summary_keys = [*self.data_keys, *self.context_keys]

    def sample(
        self,
        batch_size: int,
        num_steps: int,
        include_fixed: bool = False,
        tile_to_steps: bool = False,
        rng: np.random.Generator | None = None,
        apply_missing: bool = True,
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

        apply_missing : bool, optional, default: True
            Apply the configured missing process. False returns complete
            observations without missingness metadata.

        Returns
        -------
        result : dict - flat dictionary with the following entries:
            - one entry per simulated observation variable, each with
            shape (batch_size, num_steps), corrupted by
            `self.missing` if one is configured.
            - one entry per generated context variable, retaining the
            leading batch and time dimensions.
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
            - new formula targets consumed by the simulator are returned at their
              final linked values with shape (batch_size, num_steps, dim), where
              dim is 1 for scalar parameters. These fields are recorded in
              `formula_keys` and are not inference targets or summary inputs.
              Existing raw parameter, observation, and context names retain
              their original values.

            The instance attributes `local_keys`, `hyper_keys`,
            `shared_keys`, `fixed_keys`, `formula_keys`, and `data_keys` record which
            keys belong to which group.

        Raises
        ------
        ValueError
            If required parameters are missing from the prior or have
            invalid shapes.
        """
        # Sample parameters
        generated_context, contexts = self._generate_context(batch_size=batch_size, num_steps=num_steps)
        prior_draws = self.prior.sample(batch_size=batch_size, num_steps=num_steps)
        local_params = prior_draws["local_params"]
        deterministic_params = prior_draws.get("deterministic_params", {})
        shared_params = prior_draws.get("shared_params", {})
        fixed_params = prior_draws.get("fixed_params", {})

        # Combine parameter dictionaries
        combined_params = dict(local_params)
        combined_params.update(deterministic_params)
        combined_params.update(shared_params)

        # Keep every fixed draw available to the formula. Parameters
        # not consumed by the simulator are ignored by `_ordered_model_args`.
        combined_params.update(fixed_params)

        model_params, simulator_context = self._resolve_parameters(combined_params, contexts)
        ordered_params = self._ordered_model_args(
            model_params,
            batch_size=batch_size,
            num_steps=num_steps,
            missing_context="prior and has no default",
        )

        # Capture final linked formula parameters in the same shape as local trajectories.
        formula_params = {
            name: np.asarray(value).reshape(batch_size, num_steps, -1).copy()
            for name, value in zip(self.param_order, ordered_params)
            if name in self.formula_keys
        }

        # Run simulator
        model_output = self._call_simulator(ordered_params, simulator_context)
        sim_data = self._reshape_model_output(model_output, batch_size, num_steps, expected_data_keys=self.data_keys)

        # Apply contamination augmentation, if configured
        sim_data, contamination_extra = self._apply_contamination(sim_data, rng, model_params.get("p_contaminated"))
        if isinstance(self.contamination, RandomChoiceContamination) and self.contamination.infer:
            contamination_extra.pop("p_contaminated", None)  # Keep the raw inference target.
        self._exclude_nuisance(prior_draws)

        # Apply missingness augmentation, if configured
        missing_mask, missing_extra = None, {}
        if apply_missing:
            sim_data, missing_mask, missing_extra = self._apply_missing(sim_data, rng, model_params.get("p_missing"))

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
        overlapping_context_keys = set(generated_context) & set(result)
        if overlapping_context_keys:
            raise ValueError(
                f"Generated context keys conflict with model output keys: {sorted(overlapping_context_keys)}"
            )
        result.update(generated_context)
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
        collisions = set(formula_params) & result.keys()
        if collisions:
            raise ValueError(f"Formula parameter names conflict with returned fields: {sorted(collisions)}")
        result.update(formula_params)

        return result

    def plot_time_varying_prior(
        self,
        num_steps: int = 200,
        num_trajectories: int = 20,
        num_cols: int | None = None,
        marginal: bool = True,
        dist_type: Literal["hist", "kde", "both"] = "hist",
        num_bins: int | None = None,
        dist_alpha: float = 1.0,
        alpha: float = 0.5,
        color: str = BASE_COLOR,
        title_fontsize: int = TITLE_FONTSIZE,
        label_fontsize: int = LABEL_FONTSIZE,
        tick_fontsize: int = TICK_FONTSIZE,
        figsize: tuple[float, float] | None = None,
    ) -> Figure:
        """Plot raw time-varying inference targets, without resolving formulas or context.

        Parameters
        ----------
        num_steps        : int, optional, default: 200
            Number of time steps to sample per trajectory.
        num_trajectories : int, optional, default: 20
            Number of trajectories to draw.
        num_cols         : int or None, optional, default: None
            Number of panel columns. If None, uses the compact dynamic layout.
        marginal         : bool, optional, default: True
            Whether to display a marginal distribution beside each trajectory.
        dist_type        : {"hist", "kde", "both"}, optional, default: "hist"
            Distribution type used for marginal panels.
        num_bins         : int or None, optional, default: None
            Number of histogram bins. If None, Seaborn selects the bins.
        dist_alpha       : float, optional, default: 1.0
            Opacity of marginal distributions.
        alpha            : float, optional, default: 0.5
            Opacity of individual trajectories.
        color            : str, optional, default: BASE_COLOR
            Color used for trajectories and marginal distributions.
        title_fontsize   : int, optional, default: 22
            Font size for panel titles.
        label_fontsize   : int, optional, default: 18
            Font size for axis labels and the figure legend.
        tick_fontsize    : int, optional, default: 16
            Font size for tick labels.
        figsize          : tuple of two floats or None, optional, default: None
            Explicit figure size in inches.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The generated figure.
        """
        samples = self._sample_inference_prior(batch_size=num_trajectories, num_steps=num_steps)
        local_params = samples["local_params"]
        return plot_time_varying_prior(
            local_params=local_params,
            param_bounds={},
            num_cols=num_cols,
            marginal=marginal,
            dist_type=dist_type,
            num_bins=num_bins,
            dist_alpha=dist_alpha,
            alpha=alpha,
            color=color,
            title_fontsize=title_fontsize,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            figsize=figsize,
        )

    def plot_time_invariant_prior(
        self,
        num_draws: int = 1000,
        num_steps: int = 1,
        dist_type: Literal["hist", "kde", "both"] = "hist",
        num_bins: int | None = None,
        dist_alpha: float | None = None,
        color: str = BASE_COLOR,
        num_cols: int | None = None,
        title_fontsize: int = TITLE_FONTSIZE,
        label_fontsize: int = LABEL_FONTSIZE,
        tick_fontsize: int = TICK_FONTSIZE,
        figsize: tuple[float, float] | None = None,
    ) -> Figure:
        """Plot marginal distributions for time-invariant prior parameters.

        Parameters
        ----------
        num_draws      : int, optional, default: 1000
            Number of draws used to sample `hyper_params` and `shared_params`.
        dist_type      : {"hist", "kde", "both"}, optional, default: "both"
            Distribution plot type.
        num_bins       : int or None, optional, default: None
            Number of histogram bins. If None, Seaborn selects the bins.
        dist_alpha     : float or None, optional, default: None
            Opacity of parameter distributions. If None, uses 1.0 for one
            distribution and 0.5 for overlays.
        color          : str, optional, default: BASE_COLOR
            Color used for non-mixture distributions.
        num_cols       : int or None, optional, default: None
            Number of panel columns. If None, uses up to four columns.
        title_fontsize : int, optional, default: 22
            Font size for panel titles.
        label_fontsize : int, optional, default: 18
            Font size for axis labels.
        tick_fontsize  : int, optional, default: 16
            Font size for tick labels and legends.
        figsize        : tuple of two floats or None, optional, default: None
            Explicit figure size in inches.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The generated figure.
        """
        samples = self._sample_inference_prior(batch_size=num_draws, num_steps=num_steps)
        return plot_time_invariant_prior(
            hyper_params=samples["hyper_params"],
            shared_params=samples["shared_params"],
            mixture_names={name: obj.names for name, obj in self.prior.params.items() if hasattr(obj, "names")},
            dist_type=dist_type,
            num_bins=num_bins,
            dist_alpha=dist_alpha,
            color=color,
            num_cols=num_cols,
            title_fontsize=title_fontsize,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            figsize=figsize,
        )

    def plot_joint_prior(
        self,
        num_steps: int = 200,
        num_trajectories: int = 20,
        num_draws: int = 1000,
        marginal: bool = True,
        dist_type: Literal["hist", "kde", "both"] = "hist",
        num_bins: int | None = None,
        dist_alpha: float | None = None,
        color: str = BASE_COLOR,
        title_fontsize: int = TITLE_FONTSIZE,
        label_fontsize: int = LABEL_FONTSIZE,
        tick_fontsize: int = TICK_FONTSIZE,
        alpha: float = 0.5,
        figsize: tuple[float, float] | None = None,
    ) -> Figure:
        """Plot raw stochastic and deterministic trajectories with invariant priors.

        Deterministic trajectories are derived from their sampled curve
        coefficients and displayed alongside stochastic local trajectories.
        All trajectories remain on their raw transition scale; formulas,
        links, and simulator context are not resolved for this plot.

        Parameters
        ----------
        num_steps        : int, optional, default: 200
            Number of time steps for local trajectory sampling.
        num_trajectories : int, optional, default: 20
            Number of stochastic and deterministic trajectories to plot.
        num_draws        : int, optional, default: 1000
            Number of draws used for time-invariant parameter sampling.
        marginal         : bool, optional, default: True
            Whether to display a marginal distribution beside trajectories.
        dist_type        : {"hist", "kde", "both"}, optional, default: "hist"
            Distribution type used for marginal panels.
        num_bins         : int or None, optional, default: None
            Number of histogram bins. If None, Seaborn selects the bins.
        dist_alpha       : float or None, optional, default: None
            Opacity of all marginal and time-invariant distributions. If None,
            uses 1.0 for one distribution and 0.5 for overlays.
        color            : str, optional, default: BASE_COLOR
            Color used for trajectories and distributions.
        title_fontsize   : int, optional, default: 22
            Font size for panel titles.
        label_fontsize   : int, optional, default: 18
            Font size for row labels and the figure legend.
        tick_fontsize    : int, optional, default: 16
            Font size for tick labels.
        alpha            : float, optional, default: 0.5
            Opacity of individual trajectories.
        figsize          : tuple of two floats or None, optional, default: None
            Explicit figure size in inches.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The generated figure.
        """
        samples = self._sample_inference_prior(
            batch_size=num_draws,
            num_steps=num_steps,
            include_deterministic=True,
        )
        all_local_params = {**samples["local_params"], **samples["deterministic_params"]}
        local_params = {k: v[:num_trajectories] for k, v in all_local_params.items()}
        return plot_joint_prior(
            local_params=local_params,
            hyper_params=samples["hyper_params"],
            shared_params=samples["shared_params"],
            param_bounds={},
            mixture_names={name: obj.names for name, obj in self.prior.params.items() if hasattr(obj, "names")},
            hyper_param_groups=samples["hyper_param_groups"],
            marginal=marginal,
            dist_type=dist_type,
            num_bins=num_bins,
            dist_alpha=dist_alpha,
            color=color,
            title_fontsize=title_fontsize,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            alpha=alpha,
            figsize=figsize,
        )

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
        apply_missing: bool = False,
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

        apply_missing : bool, optional, default: False
            Apply configured missingness to the plotted observations. By
            default, show complete observations, as in posterior resimulation.

        Returns
        -------
        fig : plt.Figure - the figure containing the requested plot
        """
        sample = self.sample(batch_size=batch_size, num_steps=num_steps, apply_missing=apply_missing)
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

    def get_fixed_params(self) -> Dict[str, np.ndarray]:
        """Return deterministic fixed parameters from the prior for simulator simulation.

        Draws a single pilot sample from `self.prior` and keeps only the
        fixed-parameter entries, including auxiliary regression coefficients.

        Returns
        -------
        fixed_params : dict of np.ndarray - mapping from parameter name
            to its raw fixed value, including regression coefficients
        """
        prior_draws = self.prior.sample(batch_size=1, num_steps=1)
        fixed_params = prior_draws.get("fixed_params", {})
        return {name: np.asarray(value) for name, value in fixed_params.items()}

    def simulate_from_parameters(
        self,
        params: Dict[str, np.ndarray],
        batch_size: int,
        num_steps: int,
        context: Mapping[str, Any] | None = None,
        rng: np.random.Generator | None = None,
        apply_missing: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Simulate outputs and contamination for given raw parameter values.

        Parameters
        ----------
        params     : dict of np.ndarray
            Parameter values to simulate from, keyed by simulator parameter
            name. See `_prepare_flat_params` for the accepted shapes.
        batch_size : int
            Number of independent simulation batches.
        num_steps  : int
            Number of time steps per trajectory.
        context    : Mapping or pandas.DataFrame or None, optional, default: None
            Context for this simulation call. A mapping may contain already
            batched arrays with shape ``(batch_size, num_steps, ...)`` or one
            fixed trial sequence with shape ``(num_steps, ...)``. A DataFrame
            is interpreted as one fixed sequence and repeated across batches.
            If omitted, the model's configured context source is used.
        rng : np.random.Generator or None
            Generator used for contamination and missingness masks.
        apply_missing : bool, optional, default: False
            Apply configured missingness with a fresh nuisance probability
            from JointPrior. By default, return complete observations.

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
        if context is None:
            contexts = self._sample_context(batch_size, num_steps)
        else:
            raw_context = self._coerce_fixed_context(context, batch_size, num_steps, allow_batched=True)
            contexts = self._split_context(raw_context)
        raw_params = dict(params)
        if isinstance(self.contamination, RandomChoiceContamination):
            if not self.contamination.infer:
                nuisance = JointPrior(p_contaminated=self.prior.params["p_contaminated"]).sample(batch_size, num_steps)
                for group in ("local_params", "deterministic_params", "shared_params", "fixed_params"):
                    raw_params.update(nuisance[group])
            elif "p_contaminated" not in raw_params:
                specification = self.prior.params["p_contaminated"]
                if not np.isscalar(specification):
                    raise ValueError("Posterior p_contaminated is required when infer=True.")
                raw_params["p_contaminated"] = specification
        if apply_missing and isinstance(self.missing, RandomMissingProcess):
            nuisance = JointPrior(p_missing=self.prior.params["p_missing"]).sample(batch_size, num_steps)
            for group in ("local_params", "deterministic_params", "shared_params", "fixed_params"):
                raw_params.update(nuisance[group])
        combined_params, simulator_context = self._resolve_parameters(raw_params, contexts)

        ordered_params = self._ordered_model_args(
            combined_params,
            batch_size=batch_size,
            num_steps=num_steps,
            missing_context="params and has no default",
        )

        model_output = self._call_simulator(ordered_params, simulator_context)
        sim_data = self._reshape_model_output(model_output, batch_size, num_steps, expected_data_keys=self.data_keys)
        sim_data, _ = self._apply_contamination(sim_data, rng, combined_params.get("p_contaminated"))
        if apply_missing:
            sim_data, _, _ = self._apply_missing(sim_data, rng, combined_params.get("p_missing"))
        return sim_data

    @staticmethod
    def _context_names(names, selector):
        if isinstance(names, str) or not isinstance(names, Sequence):
            raise TypeError(f"{selector} must be a sequence of context variable names.")
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError(f"{selector} names must be non-empty strings.")
        return tuple(dict.fromkeys(names))

    def _split_context(self, context):
        """Route one context draw to regression and simulator consumers."""
        if not isinstance(context, Mapping):
            raise TypeError("Context generators must return a mapping of named variables.")
        requested = {"design_context": self.design_context, "simulator_context": self.simulator_context}
        missing = set(self.design_context) | set(self.simulator_context)
        missing -= context.keys()
        if missing:
            raise KeyError(f"Context variables requested by the model but not generated: {sorted(missing)}")
        return {consumer: {name: context[name] for name in names} for consumer, names in requested.items()}

    def _generate_context(
        self,
        batch_size: int,
        num_steps: int,
        pilot: bool = False,
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        """Generate context and split it for the model's consumers."""
        if self.context is None:
            return {}, {
                "simulator_context": {},
                "design_context": {},
            }

        if isinstance(self.context, ContextSimulator):
            context = self.context.sample(batch_size=batch_size, num_steps=num_steps)
        else:
            context = self._coerce_fixed_context(self.context, batch_size, num_steps, pilot=pilot)
        return context, self._split_context(context)

    def _sample_context(self, batch_size: int, num_steps: int) -> dict[str, dict[str, Any]]:
        """Generate and split context when the raw output is not needed."""
        _, contexts = self._generate_context(batch_size, num_steps)
        return contexts

    @staticmethod
    def _coerce_fixed_context(
        context: Mapping[str, Any] | pd.DataFrame,
        batch_size: int,
        num_steps: int,
        pilot: bool = False,
        allow_batched: bool = False,
    ) -> dict[str, np.ndarray]:
        """Repeat fixed trial-level context across simulation batches."""
        if isinstance(context, pd.DataFrame):
            if not context.columns.is_unique:
                raise ValueError("Fixed context DataFrame columns must be unique.")
            values = {column: context[column].to_numpy() for column in context.columns}
        elif isinstance(context, Mapping):
            values = dict(context)
        else:
            raise TypeError("Fixed context must be a mapping or pandas DataFrame.")

        if any(not isinstance(name, str) for name in values):
            raise TypeError("Context variable names must be strings.")

        result = {}
        for name, value in values.items():
            array = np.asarray(value)
            if array.ndim == 0:
                result[name] = np.full((batch_size, num_steps), array.item(), dtype=array.dtype)
                continue
            if allow_batched and array.ndim >= 2 and array.shape[:2] == (batch_size, num_steps):
                result[name] = array
                continue
            if array.shape[0] == 0:
                raise ValueError(f"Fixed context variable {name!r} cannot be empty.")
            if pilot:
                array = array[:1]
            elif array.shape[0] != num_steps:
                raise ValueError(
                    f"Fixed context variable {name!r} must have {num_steps} trial rows, got {array.shape[0]}."
                )
            result[name] = np.broadcast_to(array[None, ...], (batch_size, *array.shape))
        return result

    @staticmethod
    def _accepts_context(callable_: Callable) -> bool:
        """Return whether a callable accepts a ``context`` keyword."""
        signature = inspect.signature(callable_)
        return "context" in signature.parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
        )

    def _apply_links(self, parameters):
        resolved = dict(parameters)
        links = getattr(self, "link_function", {})
        targets = links if isinstance(links, Mapping) else self.param_order
        for name in targets:
            if name not in resolved:
                continue  # Simulator defaults are handled separately.
            link = links[name] if isinstance(links, Mapping) else links
            resolved[name] = link(resolved[name])
        return resolved

    def _resolve_parameters(self, parameters, contexts):
        """Resolve raw coefficients, bind parameter context, then apply links once."""
        resolved = self._resolve_formula(dict(parameters), contexts["design_context"])
        resolved, simulator_context = self._apply_simulator_context(resolved, contexts["simulator_context"])
        return self._apply_links(resolved), simulator_context

    def sample_prior(self, batch_size=20, num_steps=200, context=None):
        """Draw raw coefficient groups and derived cognitive parameters without simulating.

        ``model_params`` contains linked parameters shaped (batch, steps).
        ``model_time_varying_keys`` identifies trial-level derived parameters.
        Other groups retain the raw inference scale. Explicit context may be
        supplied to inspect the induced prior under a particular design.
        """
        if context is None:
            contexts = self._sample_context(batch_size, num_steps)
        else:
            raw = self._coerce_fixed_context(context, batch_size, num_steps, allow_batched=True)
            contexts = self._split_context(raw)
        draws = self.prior.sample(batch_size=batch_size, num_steps=num_steps)
        combined = {}
        for group in ("local_params", "deterministic_params", "shared_params", "fixed_params"):
            combined.update(draws.get(group, {}))
        resolved, _ = self._resolve_parameters(combined, contexts)
        # Include defaults for inspection, but do not present them as prior draws.
        draws["model_default_keys"] = [name for name in self.param_order if name not in resolved]
        for name in self.param_order:
            if name not in resolved and self.signature.parameters[name].default is not inspect.Parameter.empty:
                resolved[name] = self._apply_links({name: self.signature.parameters[name].default})[name]
        cognitive = {name: resolved[name] for name in self.param_order if name in resolved}
        flat = self._prepare_flat_params(cognitive, batch_size, num_steps)
        draws["model_params"] = {
            name: value.reshape(batch_size, num_steps, *value.shape[1:]) for name, value in flat.items()
        }
        draws["model_time_varying_keys"] = [
            name
            for name, value in cognitive.items()
            if np.asarray(value).ndim >= 2 and np.asarray(value).shape[1] == num_steps
        ]
        self._exclude_nuisance(draws)
        return draws

    def _exclude_nuisance(self, draws):
        for group in ("local_params", "deterministic_params", "hyper_params", "shared_params", "fixed_params"):
            for key in self._nuisance_keys:
                draws.get(group, {}).pop(key, None)

    def _sample_inference_prior(self, batch_size, num_steps, include_deterministic=False):
        """Draw raw prior groups without formulas, links, or simulator context.

        Deterministic trajectories are derived rather than inferred directly,
        so they are omitted unless ``include_deterministic`` is True.
        """
        draws = self.prior.sample(batch_size=batch_size, num_steps=num_steps)
        groups = dict(self.prior._last_hyper_param_groups)
        self._exclude_nuisance(draws)
        for name in list(groups):
            if name in self._nuisance_keys:
                groups.pop(name)
        if not include_deterministic:
            draws["deterministic_params"] = {}
        draws["hyper_param_groups"] = groups
        return draws

    def _resolve_formula(
        self,
        parameters: Dict[str, np.ndarray],
        context: Mapping[str, Any],
    ) -> Dict[str, np.ndarray]:
        """Resolve model parameters through the optional formula."""
        if self.formula is None:
            return parameters

        # Posterior scalar trajectories use (batch, steps, 1), whereas
        # design context and prior trajectories use (batch, steps).
        parameters = {
            name: np.asarray(value)[..., 0]
            if np.asarray(value).ndim == 3 and np.asarray(value).shape[-1] == 1
            else value
            for name, value in parameters.items()
        }
        resolver = getattr(self.formula, "resolve", self.formula)
        if not callable(resolver):
            raise TypeError("formula must be callable or provide a callable resolve method.")
        resolved = resolver(parameters=parameters, context=context)
        if not isinstance(resolved, Mapping):
            raise TypeError("formula must return a mapping of simulator parameters.")
        return dict(resolved)

    def _call_simulator(self, ordered_params: list, context: Mapping[str, Any]) -> Mapping[str, np.ndarray]:
        """Call the simulator, forwarding its mapped context when supported."""
        if self._accepts_context(self.simulator):
            return self.simulator(*ordered_params, context=context)
        if context:
            raise TypeError("simulator_context was supplied, but simulator does not accept a context keyword argument.")
        return self.simulator(*ordered_params)

    def _apply_simulator_context(
        self,
        parameters: Dict[str, np.ndarray],
        context: Mapping[str, Any],
    ) -> tuple[Dict[str, np.ndarray], dict[str, Any]]:
        """Bind context variables that match simulator arguments.

        This lets context generators provide conventional simulator inputs
        such as ``correct_idx`` for :func:`sample_rdm`. Any remaining context
        is forwarded as a ``context`` keyword to simulators that support it.
        """
        parameter_context = {name: value for name, value in context.items() if name in self.param_order}
        passthrough_context = {name: value for name, value in context.items() if name not in self.param_order}
        return {**parameters, **parameter_context}, passthrough_context

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
            ordered_params.append(self._apply_links({name: default})[name])

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

    def _infer_data_keys(self, prior_draws: dict, contexts: Mapping[str, Mapping[str, Any]] | None = None) -> list[str]:
        """Infer observation names from a one-step simulator call."""
        combined_params = dict(prior_draws.get("local_params", {}))
        combined_params.update(prior_draws.get("deterministic_params", {}))
        combined_params.update(prior_draws.get("shared_params", {}))

        # Formula terms can be fixed parameters even when their names are not
        # simulator arguments (e.g. ``v = v_0 + b_v * covariate``).
        combined_params.update(prior_draws.get("fixed_params", {}))

        contexts = contexts or self._sample_context(batch_size=1, num_steps=1)
        model_params, simulator_context = self._resolve_parameters(combined_params, contexts)
        sampled_keys = set().union(
            *(
                prior_draws.get(group, {}).keys()
                for group in ("local_params", "deterministic_params", "shared_params", "hyper_params", "fixed_params")
            )
        )
        sampled_keys.update(key for keys in self._contamination_parameter_groups.values() for key in keys)
        self.formula_keys = [
            name
            for name in self.param_order
            if self.formula is not None
            and name in model_params
            and name not in sampled_keys
            and name not in self.simulator_context
        ]
        ordered_params = self._ordered_model_args(
            model_params,
            batch_size=1,
            num_steps=1,
            missing_context="prior",
        )
        model_output = self._call_simulator(ordered_params, simulator_context)
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
        probability: np.ndarray | float | None = None,
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

        if isinstance(self.contamination, RandomChoiceContamination):
            out = self.contamination.apply(sim_data, rng=rng, probability=probability)
        elif isinstance(self.contamination, ContaminationProcess):
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
        probability: np.ndarray | float | None = None,
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

        if isinstance(self.missing, RandomMissingProcess):
            result = self.missing.apply(sim_data, rng=rng, probability=probability)
        else:
            result = self.missing(sim_data, rng=rng) if accepts_rng else self.missing(sim_data)

        sim_data = {key: result[key] for key in self.data_keys}
        missing_mask = result["missing_mask"]
        extra = {k: v for k, v in result.items() if k not in (*self.data_keys, "missing_mask")}
        return sim_data, missing_mask, extra
