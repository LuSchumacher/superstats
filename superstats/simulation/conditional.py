"""Resolve paired parameter draws under context held constant across time."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from math import prod
from numbers import Integral

import numpy as np
import pandas as pd

from superstats.transition import DeterministicTransition
from superstats.utils.indexing import normalize_data_indices


@dataclass
class ConditionalDraws:
    """Store paired parameter draws evaluated under fixed predictor conditions.

    Attributes
    ----------
    parameters : dict of str to np.ndarray
        Parameter arrays shaped (B, S, C, T): datasets, draws, conditions, and
        time steps. In combined raw and implied views, C may differ between
        parameters; raw coefficients have one condition.
    conditions : list of dict
        Ordered fixed predictor assignments used to evaluate the draws.
    data_indices : np.ndarray
        Original dataset indices in the order of the first parameter axis.
        Prior draws use a single dataset with index zero.
    raw_groups : dict or None, optional
        Original grouped samples returned by `JointPrior.sample`, populated
        for prior evaluation. These include local, deterministic, shared,
        fixed, and hyperparameter groups.
    raw_parameters : dict or None, optional
        Coefficients before links and formulas. Evaluators flatten dataset
        and draw axes to B*S; local trajectories have shape (B*S, T), while
        scalar shared coefficients have shape (B*S,) or (B*S, 1). Fixed
        coefficients may be scalars. Used to prepare paired raw panels.
    parameter_conditions : dict of str to list of dict or None, optional
        Per-parameter condition assignments after combining raw and implied
        panels. When present, these replace `conditions` for each parameter's
        condition axis. Combined views need not retain raw sample metadata.
    """

    parameters: dict[str, np.ndarray]
    conditions: list[dict]
    data_indices: np.ndarray
    raw_groups: dict | None = None
    raw_parameters: dict | None = None
    parameter_conditions: dict | None = None


def raw_parameter_names(model, names, kind, selected=False):
    """Select raw coefficients to accompany an implied parameter view.

    Parameters
    ----------
    model : Model
        Model defining parameter groups and formula dependencies.
    names : sequence of str
        Selected implied parameter names.
    kind : {"varying", "invariant", "joint"}
        Plot type. Varying plots use temporal coefficients, invariant plots
        use shared coefficients, and joint plots use both groups.
    selected : bool, optional, default: False
        Restrict coefficients to dependencies of `names` and include directly
        selected prior parameters. Explicit invariant selections can include
        temporal dependencies for later time reduction.

    Returns
    -------
    names : list of str
        Unique raw coefficient names in model order, followed by additional
        explicit selections. Hyperparameters are not selected automatically,
        and nuisance parameters are excluded.
    """
    temporal = [*model.local_keys, *model.deterministic_keys]
    candidates = temporal if kind == "varying" else list(model.shared_keys)
    if kind == "joint" or (kind == "invariant" and selected):
        candidates = [*temporal, *model.shared_keys]
    if selected:
        dependencies = parameter_dependencies(model, names)
        candidates = [name for name in candidates if name in dependencies]
        candidates += [name for name in names if name in model.prior.params and name not in model.hyper_keys]
    return list(dict.fromkeys(name for name in candidates if name not in model._nuisance_keys))


def parameter_views(model, draws, raw_names, truth=None):
    """Combine paired raw and implied draws into separate plotting panels.

    Parameters
    ----------
    model : Model
        Model defining temporal coefficients and predictor dependencies.
    draws : ConditionalDraws
        Nonempty evaluated scalar parameters with populated `raw_parameters`.
        Parameter arrays have shape (B, S, C, T).
    raw_names : sequence of str
        Scalar raw coefficients to include. Vector parameters are handled
        separately by `_vector_views`.
    truth : ConditionalDraws or None, optional, default: None
        Evaluated targets with one draw per dataset, populated raw coefficients,
        and the same parameter names, dataset order, and conditions as `draws`.

    Returns
    -------
    views : ConditionalDraws
        Panels labeled with " [raw]" or " [implied]". Raw values have shape
        (B, S, 1, T); implied values retain only relevant conditions. A raw and
        implied array with the same name and exactly equal values share one
        unsuffixed panel. Hyperparameters remain unsuffixed.
    target_views : ConditionalDraws or None
        Targets following the panel names and condition selection decided from
        the draws, or None when no targets were supplied.

    Notes
    -----
    All views use the supplied draws without resampling or applying links again.
    Returned containers include per-parameter conditions, but do not copy the
    raw sample metadata from the input containers.
    """
    exemplar = next(iter(draws.parameters.values()))
    datasets, samples, _, steps = exemplar.shape
    temporal = set(model.local_keys) | set(model.deterministic_keys)

    def raw_values(source, name, count):
        """Reshape a raw coefficient to (B, count, 1, T) for a single panel.

        Parameters
        ----------
        source : ConditionalDraws
            Container with populated raw coefficients.
        name : str
            Scalar coefficient name.
        count : int
            Number of draws per dataset.

        Returns
        -------
        values : np.ndarray
            Raw trajectories, or shared values broadcast over time.
        """
        values = np.asarray(source.raw_parameters[name])
        if name in temporal:
            return values.reshape(datasets, count, 1, steps)
        values = np.broadcast_to(values.reshape(-1), (datasets * count,))
        return np.broadcast_to(values.reshape(datasets, count, 1, 1), (datasets, count, 1, steps))

    raw = {name: raw_values(draws, name, samples) for name in raw_names}
    implied, conditions = {}, {}
    for name, values in draws.parameters.items():
        indices, _ = condition_indices(model, draws, name)
        implied[name] = values[:, :, indices]
        conditions[name] = [draws.conditions[index] for index in indices]
    # Decide once using the estimates; targets must follow exactly the same panels.
    identical = {name for name in raw.keys() & implied.keys() if np.array_equal(raw[name], implied[name])}
    panels, scenarios, target_panels = {}, {}, {}
    for name, values in raw.items():
        label = name if name in identical else f"{name} [raw]"
        panels[label], scenarios[label] = values, [{}]
        if truth is not None:
            target_panels[label] = raw_values(truth, name, 1)
    for name, values in implied.items():
        if name in identical:
            continue
        # Hyperparameters stay on their own scale and appear only once.
        label = name if name in model.hyper_keys else f"{name} [implied]"
        panels[label], scenarios[label] = values, conditions[name]
        if truth is not None:
            indices, _ = condition_indices(model, draws, name)
            target_panels[label] = truth.parameters[name][:, :, indices]
    result = ConditionalDraws(panels, draws.conditions, draws.data_indices, parameter_conditions=scenarios)
    target_result = (
        ConditionalDraws(target_panels, draws.conditions, draws.data_indices, parameter_conditions=scenarios)
        if truth is not None
        else None
    )
    return result, target_result


def parameter_base_name(name):
    """Remove a plotting view suffix from a parameter name.

    Parameters
    ----------
    name : str
        Parameter label, optionally ending in " [raw]" or " [implied]".

    Returns
    -------
    name : str
        Underlying parameter name.
    """
    return name.removesuffix(" [raw]").removesuffix(" [implied]")


def parameter_display_names(names, display_names, panels):
    """Expand custom parameter labels to match raw and implied panels.

    Parameters
    ----------
    names : sequence of str
        Original selected parameter names.
    display_names : sequence of str or None
        Custom labels for either the selected parameters or the final panels.
    panels : sequence or mapping of str
        Final panel names in plotting order.

    Returns
    -------
    labels : sequence of str or None
        Labels matching the panels. Final-panel labels are returned unchanged;
        selected-parameter labels are expanded with view suffixes. Additional
        raw dependencies retain their base names. None remains None.

    Raises
    ------
    ValueError
        If the supplied labels match neither the selection nor the panel count.
    """
    if display_names is None:
        return None
    if len(display_names) == len(panels):
        return display_names
    if len(display_names) != len(names):
        raise ValueError("variable_names must have one name per selected or plotted parameter.")
    aliases = dict(zip(names, display_names))
    return [
        aliases.get(parameter_base_name(name), parameter_base_name(name)) + name[len(parameter_base_name(name)) :]
        for name in panels
    ]


def parameter_dependencies(model, names):
    """Follow formula targets back to their coefficients and predictors.

    Parameters
    ----------
    model : Model
        Model with optional formula dependency metadata.
    names : sequence of str
        Parameter names whose dependencies should be traversed.

    Returns
    -------
    dependencies : set of str
        Selected names and their transitive dependencies. For a callable
        formula without dependency metadata, conservatively return all prior
        parameter and design-context names.
    """
    dependencies = getattr(model.formula, "dependencies", None)
    if dependencies is None and model.formula is not None:
        return set(model.prior.params) | set(model.design_context)
    pending, visited = list(names), set()
    while pending:
        name = pending.pop()
        if name not in visited:
            visited.add(name)
            pending.extend((dependencies or {}).get(name, ()))
    return visited


def varying_parameter_names(model, variable_keys=None):
    """Select implied parameters for a time-varying plot.

    Parameters
    ----------
    model : Model
        Model defining local processes, deterministic curves, and formulas.
    variable_keys : str, sequence of str, or None, optional, default: None
        Explicit parameter selection, validated and returned without enforcing
        temporal variation. If None, replace consumed temporal coefficients
        with their simulator-facing formula targets and retain other processes.

    Returns
    -------
    names : list of str
        Unique parameter names in plotting order.

    Raises
    ------
    ValueError
        If the default selection contains no time-varying parameters, or an
        explicit selection is empty or contains unknown names.
    """
    if variable_keys is not None:
        return _parameter_names(model, variable_keys)
    temporal = [*model.local_keys, *model.deterministic_keys]
    targets = [name for name in model.param_order if name in getattr(model.formula, "targets", ())]
    consumed = parameter_dependencies(model, targets) - set(targets)
    names = [name for name in temporal if name not in consumed]
    names += [name for name in targets if parameter_dependencies(model, [name]) & set(temporal)]
    if not names:
        raise ValueError("No time-varying parameters found. Select variable_keys or use an invariant plot.")
    return list(dict.fromkeys(names))


def invariant_parameter_names(model, variable_keys=None):
    """Select shared parameters, hyperparameters, and stationary formula targets.

    Parameters
    ----------
    model : Model
        Model defining parameter groups and formula dependencies.
    variable_keys : str, sequence of str, or None, optional, default: None
        Explicit parameter selection. If None, use inferred hyperparameters,
        shared coefficients, and simulator-facing formula targets without
        temporal dependencies. Explicit names may include dynamic quantities
        for later time reduction.

    Returns
    -------
    names : list of str
        Unique parameter names in plotting order. The default may be empty.
    """
    if variable_keys is not None:
        return _parameter_names(model, variable_keys)
    temporal = set(model.local_keys) | set(model.deterministic_keys)
    names = [*model.hyper_keys, *model.shared_keys]
    names += [
        name
        for name in model.param_order
        if name in getattr(model.formula, "targets", ()) and not parameter_dependencies(model, [name]) & temporal
    ]
    return list(dict.fromkeys(names))


def parameter_bounds(model, names):
    """Find link-function bounds on the final plotted quantities.

    Parameters
    ----------
    model : Model
        Model defining latent links, formula links, and simulator context.
    names : sequence or mapping of str
        Plotted parameter names, optionally carrying view suffixes.

    Returns
    -------
    bounds : dict of str to tuple
        Bounds keyed by the original panel names. Formula targets use only
        their formula-link bounds; other parameters use latent-link bounds.
        Raw panels, simulator-context variables, and links without declared
        bounds are omitted. Bounds on formula terms are not propagated.
    """
    result = {}
    targets = set(getattr(model.formula, "targets", ()))
    for label in names:
        if label.endswith(" [raw]"):
            continue
        name = parameter_base_name(label)
        if name in model.simulator_context:
            continue
        links = model.formula_link_function if name in targets else model.latent_link_function
        link = links.get(name) if isinstance(links, Mapping) else links
        if getattr(link, "bounds", None) is not None:
            result[label] = link.bounds
    return result


def condition_indices(model, draws, name):
    """Collapse conditions that differ only in predictors unrelated to a parameter.

    Parameters
    ----------
    model : Model
        Model providing formula and context dependencies.
    draws : ConditionalDraws
        Evaluated draws with global or per-parameter condition assignments.
    name : str
        Parameter or panel name. Raw panels have no predictor dependencies.

    Returns
    -------
    indices : list of int
        First index of each distinct relevant predictor assignment, preserving
        the order of the parameter's condition axis.
    labels : list of str
        Comma-separated predictor assignments for the selected indices.
        Parameters without relevant predictors receive one empty label.
    """
    if draws.parameter_conditions is not None:
        conditions = draws.parameter_conditions[name]
        relevant = _context_dependencies(model, [parameter_base_name(name)]) if not name.endswith(" [raw]") else set()
    else:
        conditions = draws.conditions
        relevant = _context_dependencies(model, [name])
    seen, indices, labels = [], [], []
    for index, condition in enumerate(conditions):
        selected = {key: value for key, value in condition.items() if key in relevant}
        if selected not in seen:
            seen.append(selected)
            indices.append(index)
            labels.append(
                ", ".join(
                    f"{key}={value:g}" if isinstance(value, (int, float)) else f"{key}={value}"
                    for key, value in selected.items()
                )
            )
    return indices, labels


def _vector_parameter(model, name):
    """Identify vector-valued mixture weights and Dirichlet parameters.

    Parameters
    ----------
    model : Model
        Model containing the prior specifications.
    name : str
        Unsuffixed parameter name.

    Returns
    -------
    is_vector : bool
        True for names ending in "_mixture_weights" or Dirichlet priors.
    """
    return name.endswith("_mixture_weights") or getattr(model.prior.params.get(name), "dist", None) == "dirichlet"


def _vector_views(name, raw, implied, scale, raw_target=None, implied_target=None):
    """Assign panel names to raw and implied vector parameters and targets.

    Parameters
    ----------
    name : str
        Unsuffixed vector parameter name.
    raw, implied : np.ndarray
        Paired vector samples before and after latent links, with matching axes.
    scale : {"model", "both"}
        Whether to include separate raw and implied views.
    raw_target, implied_target : np.ndarray or None, optional, default: None
        Matching vector targets before and after latent links.

    Returns
    -------
    samples : dict of str to np.ndarray
        One unsuffixed implied panel, or separate suffixed panels when `scale`
        is "both" and sample values differ.
    targets : dict of str to np.ndarray or None
        Target entries using the same naming decision as the samples. Entries
        remain None when their corresponding targets were not supplied.
    """
    if scale != "both" or np.array_equal(raw, implied):
        return {name: implied}, {name: implied_target}
    return (
        {f"{name} [raw]": raw, f"{name} [implied]": implied},
        {f"{name} [raw]": raw_target, f"{name} [implied]": implied_target},
    )


def prior_plot_samples(
    model, num_draws, num_steps, conditions=None, context=None, variable_keys=None, kind="joint", scale="model"
):
    """Prepare paired prior samples for trajectory and distribution layouts.

    Parameters
    ----------
    model : Model
        Model providing the joint prior, parameter groups, links, and formulas.
    num_draws : int
        Positive number of prior draws. A single joint sample supplies every
        selected parameter, condition, and raw or implied view.
    num_steps : int
        Positive trajectory length used for sampling and context references.
    conditions : mapping, sequence of mappings, or None, optional, default: None
        Fixed predictor grid or explicit scalar assignments. If None, use the
        automatic conditions described in `_make_conditions`.
    context : mapping or pandas.DataFrame or None, optional, default: None
        Reference predictor values for automatic conditions and omitted
        predictors. If None, use the model's configured context source.
    variable_keys : str, sequence of str, or None, optional, default: None
        Selected parameter names. Defaults follow the requested plot kind.
        Explicit selections also restrict included hyperparameters to selected
        names and the hyperparameters owned by their dependencies.
    kind : {"joint", "varying", "invariant"}, optional, default: "joint"
        Plot layout to prepare. Invariant layouts reject temporal quantities.
    scale : {"model", "both"}, optional, default: "model"
        Evaluate implied parameters alone, or add paired raw coefficient panels.
        The public plot methods handle their raw-only path separately.

    Returns
    -------
    local : dict of str to np.ndarray
        Trajectories shaped (S, T), or (S, C, T) for multiple conditions.
    hyper : dict of str to np.ndarray
        Inferred hyperparameter draws on their original scale. In combined
        views, each is associated with its underlying raw process once.
    shared : dict of str to np.ndarray
        Scalar draws shaped (S,), conditional draws shaped (S, C), or vector
        draws shaped (S, K), where K is the number of vector components.
    labels : dict of str to list of str
        Condition labels for parameters with more than one relevant condition.
    owners : dict of str to list of str
        Hyperparameter names grouped under the trajectory or process panel
        that owns them.

    Raises
    ------
    ValueError
        If sample sizes or selections are invalid, invariant selections depend
        on temporal parameters, conditions are invalid, or no parameters can
        be plotted.

    Notes
    -----
    This function samples parameters without calling the observation simulator.
    Nuisance parameters are excluded from plotted groups.
    """
    temporal = set(model.local_keys) | set(model.deterministic_keys)
    if kind == "invariant":
        varying = []
    else:
        try:
            varying = varying_parameter_names(model, variable_keys)
        except ValueError:
            if kind != "joint":
                raise
            varying = []
    invariant = (
        [] if kind == "varying" else invariant_parameter_names(model, variable_keys if kind == "invariant" else None)
    )
    if kind == "joint" and variable_keys is not None:
        selected = _parameter_names(model, variable_keys)
        varying = [name for name in selected if parameter_dependencies(model, [name]) & temporal]
        invariant = [name for name in selected if name not in varying]
    if any(parameter_dependencies(model, [name]) & temporal for name in invariant):
        raise ValueError("Time-varying parameters belong in plot_time_varying_prior, not an invariant plot.")
    names = list(dict.fromkeys([*varying, *(name for name in invariant if not _vector_parameter(model, name))]))
    if names:
        evaluated = prior_parameters(
            model, names, num_trajectories=num_draws, num_steps=num_steps, conditions=conditions, context=context
        )
        groups = evaluated.raw_groups
    else:
        _positive_integer(num_draws, "num_draws")
        _positive_integer(num_steps, "num_steps")
        groups = model.prior.sample(batch_size=num_draws, num_steps=num_steps)
        evaluated = ConditionalDraws({}, [{}], np.array([0]), groups)
    hyper = {name: value for name, value in groups["hyper_params"].items() if name not in model._nuisance_keys}
    if variable_keys is not None:
        dependencies = parameter_dependencies(model, [*varying, *invariant])
        owned = {
            key for owner, keys in model.prior._last_hyper_param_groups.items() if owner in dependencies for key in keys
        }
        hyper = {name: value for name, value in hyper.items() if name in invariant or name in owned}
    if scale == "both" and names:
        raw_names = raw_parameter_names(model, [*varying, *invariant], kind=kind, selected=variable_keys is not None)
        raw_names = [name for name in raw_names if not _vector_parameter(model, name)]
        evaluated, _ = parameter_views(model, evaluated, raw_names)
        varying = [
            name
            for name in evaluated.parameters
            if kind == "varying"
            or (name.endswith(" [raw]") and parameter_base_name(name) in temporal)
            or (not name.endswith(" [raw]") and parameter_base_name(name) in varying)
        ]
        invariant = [name for name in evaluated.parameters if name not in varying] + [
            name for name in invariant if _vector_parameter(model, name)
        ]
    shared, local, labels = {}, {}, {}
    for name in [*varying, *invariant]:
        if name in hyper:
            continue
        if _vector_parameter(model, name):
            linked = model._apply_links(groups["shared_params"], model.latent_link_function, groups["shared_params"])
            values, _ = _vector_views(name, groups["shared_params"][name], linked[name], scale=scale)
            shared.update(values)
            continue
        indices, names_for_conditions = condition_indices(model, evaluated, name)
        values = evaluated.parameters[name][0][:, indices, :]
        if name in varying:
            local[name] = values if len(indices) > 1 else values[:, 0, :]
        else:
            shared[name] = values[:, :, 0] if len(indices) > 1 else values[:, 0, 0]
        if len(indices) > 1:
            labels[name] = names_for_conditions
    owners = {}
    for name in varying:
        if scale == "both" and name.endswith(" [implied]"):
            owners[name] = []
            continue
        dependencies = parameter_dependencies(model, [parameter_base_name(name)])
        owners[name] = list(
            dict.fromkeys(
                key
                for owner, keys in model.prior._last_hyper_param_groups.items()
                if owner in dependencies
                for key in keys
                if key in hyper
            )
        )
    owned = {key for keys in owners.values() for key in keys}
    for owner, keys in model.prior._last_hyper_param_groups.items():
        remaining = [key for key in keys if key in hyper and key not in owned]
        if remaining:
            owners[owner] = remaining
    if not local and not shared and not hyper:
        raise ValueError("No parameters available for this plot.")
    return local, hyper, shared, labels, owners


def joint_prior_groups(model, local, hyper, shared, scale="both"):
    """Group joint-prior panels by their observation-model parameter.

    Parameters
    ----------
    model : Model
        Model providing simulator parameter order and formula dependencies.
    local, hyper, shared : mapping of str to np.ndarray
        Prepared trajectories, hyperparameters, and shared distributions.
    scale : {"both", "model", "raw"}, optional, default: "both"
        View represented by the supplied samples.

    Returns
    -------
    groups : dict of str to dict of str to list of str
        Rows in observation-model order, each containing "hyper", "raw", and
        "implied" panel keys. A key present in both views denotes an identical
        view rendered once. Formula coefficients belong to their resolved
        target's row. Shared hyperparameters appear in their first owning row.
        Inferred parameters unrelated to simulator inputs retain their own rows.

    Notes
    -----
    Consumed coefficients' linked distributions are omitted when their resolved
    target is present; their raw distributions remain in combined views.
    This function groups existing samples without evaluating or resampling them.
    """
    panels = {**local, **shared}
    hyper_owners = model.prior._last_hyper_param_groups
    groups, consumed, assigned_hyper = {}, set(), set()

    def add_group(name):
        """Assign existing dependent panels to one observation-parameter row.

        Parameters
        ----------
        name : str
            Observation parameter or remaining inferred process to group.
        """
        dependencies = parameter_dependencies(model, [name])
        related = [key for key in panels if parameter_base_name(key) in dependencies]
        raw = [key for key in related if scale == "raw" or (scale == "both" and not key.endswith(" [implied]"))]
        implied = [
            key for key in related if scale != "raw" and parameter_base_name(key) == name and not key.endswith(" [raw]")
        ]
        if scale != "raw" and not implied:
            implied = [key for key in related if key not in raw and not key.endswith(" [raw]")]
        owned = list(
            dict.fromkeys(
                key
                for owner, keys in hyper_owners.items()
                if owner in dependencies
                for key in keys
                if key in hyper and key not in assigned_hyper
            )
        )
        if raw or implied or owned:
            groups[name] = {"hyper": owned, "raw": raw, "implied": implied}
            consumed.update(related)
            assigned_hyper.update(owned)

    for name in model.param_order:
        if name not in model.simulator_context:
            add_group(name)
    for key in panels:
        if key not in consumed:
            add_group(parameter_base_name(key))
    for owner, keys in hyper_owners.items():
        if any(key in hyper and key not in assigned_hyper for key in keys):
            add_group(owner)
    for key in hyper:
        if key not in assigned_hyper:
            groups[key] = {"hyper": [key], "raw": [], "implied": []}
    return groups


def _positive_integer(value, name):
    """Require a positive integer, excluding Boolean values.

    Parameters
    ----------
    value : object
        Value to validate; Python and NumPy integral types are accepted.
    name : str
        Argument name included in the error message.

    Raises
    ------
    ValueError
        If `value` is Boolean, nonintegral, or not greater than zero.
    """
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _parameter_names(model, parameter):
    """Normalize and validate names for conditional parameter evaluation.

    Parameters
    ----------
    model : Model
        Model providing prior parameters, formula targets, hyperparameters,
        and simulator-bound context names.
    parameter : str, sequence of str, or None
        Selected names. If None, use available simulator parameters in
        simulator order, excluding simulator-context variables.

    Returns
    -------
    names : list of str
        Unique selected names in their original order. Nuisance parameters
        are not available for selection.

    Raises
    ------
    TypeError
        If the selection is neither a name nor a sequence of names.
    ValueError
        If the selection is empty, includes non-string entries, or contains
        unavailable names.
    """
    available = set(model.prior.params) | set(model.hyper_keys) | set(getattr(model.formula, "targets", ()))
    available |= set(model.simulator_context) & set(model.param_order)
    available -= model._nuisance_keys
    if parameter is None:
        names = [name for name in model.param_order if name in available and name not in model.simulator_context]
    elif isinstance(parameter, str):
        names = [parameter]
    elif isinstance(parameter, Sequence):
        names = list(parameter)
    else:
        raise TypeError("parameter must be a name or a sequence of names.")
    if not names or any(not isinstance(name, str) for name in names):
        raise ValueError("Select at least one parameter name.")
    unknown = set(names) - available
    if unknown:
        raise ValueError(f"Unknown parameter names: {sorted(unknown)}")
    return list(dict.fromkeys(names))


def _context_dependencies(model, names):
    """Find design and simulator-context predictors needed by selected parameters.

    Parameters
    ----------
    model : Model
        Model providing formula dependencies and context names.
    names : sequence of str
        Parameter names to trace through formula dependencies.

    Returns
    -------
    predictors : set of str
        Relevant context variables. Callable formulas without dependency
        metadata conservatively require all design-context variables, plus
        explicitly selected simulator-context names.
    """
    dependencies = getattr(model.formula, "dependencies", None)
    if dependencies is None and model.formula is not None:
        return set(model.design_context) | (set(names) & set(model.simulator_context))
    pending = list(names)
    visited = set()
    while pending:
        name = pending.pop()
        if name not in visited:
            visited.add(name)
            pending.extend((dependencies or {}).get(name, ()))
    return visited & (set(model.design_context) | set(model.simulator_context))


def _numeric_values(reference, name):
    """Extract finite, observed numeric values for a reference predictor.

    Parameters
    ----------
    reference : mapping of str to array-like
        Predictor values and an optional "missing_mask" with the same shape
        as the selected predictor. True or 1 denotes a missing entry.
    name : str
        Predictor whose reference distribution should be extracted.

    Returns
    -------
    values : np.ndarray
        Flattened finite floating-point values after removing masked entries.

    Raises
    ------
    ValueError
        If the predictor is absent, the mask is invalid, conversion to numeric
        values fails, or no finite observed values remain.
    """
    if name not in reference:
        raise ValueError(f"Supply context or explicit conditions for predictor {name!r}.")
    values = np.asarray(reference[name])
    if "missing_mask" in reference:
        mask = np.asarray(reference["missing_mask"])
        if mask.shape != values.shape:
            raise ValueError(f"missing_mask must match context variable {name!r}.")
        if mask.dtype != bool and not np.all(np.isin(mask, (0, 1))):
            raise ValueError("missing_mask must contain only Boolean or 0/1 values.")
        values = values[~mask.astype(bool)]
    try:
        values = np.asarray(values, dtype=float).ravel()
    except (TypeError, ValueError) as error:
        raise ValueError(f"Predictor {name!r} must be numeric or declared as Categorical.") from error
    values = values[np.isfinite(values)]
    if not values.size:
        raise ValueError(f"Predictor {name!r} has no observed finite values; supply explicit conditions.")
    return values


def _reference_context(model, context, num_steps):
    """Obtain predictor reference values without simulating observations.

    Parameters
    ----------
    model : Model
        Model whose configured context source is used when `context` is None.
    context : mapping or pandas.DataFrame or None
        Explicit predictor reference values. DataFrame columns must be unique.
    num_steps : int
        Number of steps requested from a configured context simulator.

    Returns
    -------
    reference : mapping of str to array-like
        Supplied mapping, DataFrame columns converted to arrays, or a context
        sample with batch size eight. If no source exists, return an empty
        dictionary.

    Raises
    ------
    TypeError
        If explicit context is neither a mapping nor a DataFrame.
    ValueError
        If DataFrame column names are duplicated.
    """
    source = model.context_simulator if context is None else context
    if source is None:
        return {}
    if isinstance(source, pd.DataFrame):
        if not source.columns.is_unique:
            raise ValueError("Context DataFrame columns must be unique.")
        return {name: source[name].to_numpy() for name in source.columns}
    if isinstance(source, Mapping):
        return source
    if context is None:
        return source.sample(batch_size=8, num_steps=num_steps)
    raise TypeError("context must be a mapping or DataFrame.")


def _make_conditions(model, names, conditions, context, num_steps):
    """Build scalar predictor assignments held constant along each trajectory.

    Parameters
    ----------
    model : Model
        Model defining design context, simulator context, and categorical levels.
    names : sequence of str
        Selected parameters used to determine relevant predictor dependencies.
    conditions : mapping, sequence of mappings, or None
        A mapping assigns each predictor a scalar or nonempty one-dimensional
        sequence and forms their Cartesian product. A sequence of mappings
        selects explicit combinations. None selects automatic conditions.
    context : mapping or pandas.DataFrame or None
        Reference values for omitted numeric predictors and automatic choices.
        If None, use the model's configured context source.
    num_steps : int
        Reference sequence length when context must be generated.

    Returns
    -------
    scenarios : list of dict
        Unique complete scalar assignments in selection order. Automatic
        choices vary only relevant predictors: all declared categorical levels,
        both levels of observed 0/1 dummies, and mean minus one population SD,
        mean, and mean plus one SD for other numeric predictors. Numeric choices
        are clipped to the observed range and deduplicated.

    Raises
    ------
    TypeError
        If `conditions` has an unsupported type.
    ValueError
        If predictors or values are invalid, required references are missing,
        or an automatic grid would exceed 12 combinations. Explicit grids and
        explicit combinations are not subject to that limit.

    Notes
    -----
    Omitted predictors are fixed at their categorical reference level, zero
    for observed 0/1 dummies, or their numeric mean. Reference statistics exclude
    nonfinite values and entries marked missing. Formula evaluation receives
    all required design predictors, including unrelated predictors held fixed.
    """
    # Only simulator arguments that are being inspected need simulator context.
    use_formula = bool(set(names) & set(getattr(model.formula, "targets", ()))) or (
        model.formula is not None and not hasattr(model.formula, "targets")
    )
    needed = list(
        dict.fromkeys(
            [*(model.design_context if use_formula else ()), *(n for n in names if n in model.simulator_context)]
        )
    )
    relevant = _context_dependencies(model, names)
    if conditions is None:
        rows = None
        grid = {}
    elif isinstance(conditions, Mapping):
        rows = None
        grid = dict(conditions)
    elif isinstance(conditions, Sequence) and not isinstance(conditions, (str, bytes)):
        rows = list(conditions)
        if not rows or any(not isinstance(row, Mapping) for row in rows):
            raise ValueError("conditions must contain at least one mapping of predictors to scalar values.")
        grid = {}
    else:
        raise TypeError("conditions must be a predictor grid or a sequence of scalar context mappings.")

    specified = set(grid) if rows is None else set().union(*(set(row) for row in rows))
    unknown = specified - set(needed)
    if unknown:
        raise ValueError(f"Unknown condition predictors: {sorted(unknown)}")
    covered = specified if rows is None else set.intersection(*(set(row) for row in rows))
    needs_reference = set(needed) - covered - model.categorical.keys()
    reference = _reference_context(model, context, num_steps) if needs_reference else {}

    baseline = {}
    for name in needed:
        if name in model.categorical:
            baseline[name] = model.categorical[name].reference
        elif name not in covered:
            values = _numeric_values(reference, name)
            baseline[name] = 0 if set(np.unique(values)) == {0, 1} else float(values.mean())

    if conditions is None:
        for name in needed:
            if name not in relevant:
                continue
            if name in model.categorical:
                grid[name] = model.categorical[name].levels
            else:
                values = _numeric_values(reference, name)
                unique = np.unique(values)
                if set(unique) == {0, 1}:
                    grid[name] = [0, 1]
                else:
                    mean, sd = values.mean(), values.std()
                    grid[name] = np.unique(np.clip([mean - sd, mean, mean + sd], values.min(), values.max()))

    if rows is None:
        options = []
        for name, values in grid.items():
            array = np.asarray(values, dtype=object)
            if array.ndim > 1 or array.size == 0:
                raise ValueError(
                    f"Condition values for {name!r} must be a scalar or non-empty one-dimensional sequence."
                )
            options.append(array.reshape(-1).tolist())
        count = prod(len(values) for values in options)
        if conditions is None and count > 12:
            raise ValueError(f"Automatic conditions would create {count} groups. Specify conditions for this plot.")
        rows = [dict(zip(grid, values)) for values in product(*options)]

    result = []
    for row in rows:
        scenario = {**baseline, **row}
        for name, value in scenario.items():
            if np.asarray(value).ndim != 0:
                raise ValueError(f"Condition {name!r} must be scalar so it stays fixed across time.")
            if name in model.categorical:
                model.categorical[name].indices(value, name=name)
            elif not isinstance(value, (int, float, np.number, bool)) or not np.isfinite(value):
                raise ValueError(f"Condition {name!r} must be a finite numeric scalar.")
            if isinstance(value, np.generic):
                scenario[name] = value.item()
        if scenario not in result:
            result.append(scenario)
    return result


def _evaluate(model, raw, names, conditions, num_datasets, num_draws, num_steps, indices):
    """Resolve existing scalar parameter draws under each fixed condition.

    Parameters
    ----------
    model : Model
        Model supplying latent links, ordered formulas, formula links, and
        simulator-context bindings.
    raw : mapping of str to array-like
        Raw coefficients with dataset and draw axes flattened to B*S. Local
        trajectories have shape (B*S, T); shared scalars are broadcast over time.
    names : sequence of str
        Scalar parameters or formula targets to extract after evaluation.
    conditions : sequence of mappings
        Complete scalar predictor assignments, one per condition.
    num_datasets, num_draws, num_steps : int
        Dataset count B, draws per dataset S, and trajectory length T.
    indices : np.ndarray
        Original dataset indices corresponding to the flattened raw arrays.

    Returns
    -------
    evaluated : ConditionalDraws
        Resolved arrays shaped (B, S, C, T), where C is the number of conditions.

    Raises
    ------
    KeyError
        If a requested parameter is unavailable after resolution.
    ValueError
        If a selected parameter cannot be broadcast to scalar values per trial
        or contains nonnumeric or nonfinite values.

    Notes
    -----
    Latent links precede formulas and formula links follow them. Hyperparameters
    bypass these operations and retain their own scale. If no formula target
    is selected, unrelated formulas are skipped. No observations or new
    parameter draws are generated.
    """
    batch_size = num_datasets * num_draws
    hyper = {name: value for name, value in raw.items() if name in model.hyper_keys}
    raw = {name: value for name, value in raw.items() if name not in model.hyper_keys}
    use_formula = bool(set(names) & set(getattr(model.formula, "targets", ()))) or (
        model.formula is not None and not hasattr(model.formula, "targets")
    )
    evaluated = {name: [] for name in names}
    for scenario in conditions:
        context = {
            name: np.full((batch_size, num_steps), value, dtype=object if name in model.categorical else None)
            for name, value in scenario.items()
        }
        if use_formula:
            resolved, _ = model._resolve_parameters(
                raw,
                {
                    "design_context": {name: context[name] for name in model.design_context},
                    "simulator_context": {
                        name: value for name, value in context.items() if name in model.simulator_context
                    },
                },
            )
        else:
            resolved = model._apply_links(raw, model.latent_link_function, raw)
            resolved.update({name: value for name, value in context.items() if name in model.simulator_context})
        resolved.update(hyper)
        for name in names:
            if name not in resolved:
                raise KeyError(f"No draws available for parameter {name!r}.")
            values = np.asarray(resolved[name])
            if values.shape == (batch_size, num_steps, 1):
                values = values[..., 0]
            if values.shape == (batch_size,):
                values = values[:, None]
            try:
                values = np.broadcast_to(values, (batch_size, num_steps))
            except ValueError as error:
                raise ValueError(f"Parameter {name!r} must be scalar-valued per trial; got {values.shape}.") from error
            if not np.issubdtype(values.dtype, np.number) or not np.all(np.isfinite(values)):
                raise ValueError(f"Parameter {name!r} must contain finite numeric values.")
            evaluated[name].append(values.reshape(num_datasets, num_draws, num_steps))
    return ConditionalDraws({name: np.stack(values, axis=2) for name, values in evaluated.items()}, conditions, indices)


def prior_parameters(model, parameter=None, num_trajectories=20, num_steps=200, conditions=None, context=None):
    """Sample the joint prior once and resolve every draw under fixed conditions.

    Parameters
    ----------
    model : Model
        Model defining the prior, links, formulas, and predictor context.
    parameter : str, sequence of str, or None, optional, default: None
        Scalar coefficients or formula targets to evaluate. If None, select
        available simulator parameters, excluding simulator-context variables.
    num_trajectories : int, optional, default: 20
        Positive number S of joint prior draws.
    num_steps : int, optional, default: 200
        Positive number T of steps per trajectory.
    conditions : mapping, sequence of mappings, or None, optional, default: None
        Predictor grid or explicit scalar assignments held fixed across time.
        If None, choose categorical levels, observed dummy levels, or numeric
        mean and mean plus/minus one SD. See `_make_conditions` for details.
    context : mapping or pandas.DataFrame or None, optional, default: None
        Reference predictor values for automatic conditions and omitted
        predictors. If None, use the model's configured context source.

    Returns
    -------
    draws : ConditionalDraws
        Resolved scalar arrays shaped (1, S, C, T), with one dataset axis and
        C fixed conditions. Original grouped draws and raw coefficients are
        retained in `raw_groups` and `raw_parameters` for paired raw panels.

    Raises
    ------
    ValueError
        If draw counts, parameter selections, conditions, or resolved scalar
        values are invalid, or an automatic grid exceeds 12 conditions.
    KeyError
        If a requested parameter cannot be resolved from the sampled prior.

    Notes
    -----
    Every condition reuses the same joint draws. Latent links, formulas, and
    formula links follow the model's simulation order; the observation simulator
    is not called. Trajectories vary through their parameters while each
    predictor assignment stays constant over time.
    """
    _positive_integer(num_trajectories, "num_trajectories")
    _positive_integer(num_steps, "num_steps")
    names = _parameter_names(model, parameter)
    scenarios = _make_conditions(model, names, conditions, context, num_steps)
    draws = model.prior.sample(batch_size=num_trajectories, num_steps=num_steps)
    raw = {}
    for group in ("local_params", "deterministic_params", "shared_params", "fixed_params", "hyper_params"):
        raw.update(draws.get(group, {}))
    result = _evaluate(model, raw, names, scenarios, 1, num_trajectories, num_steps, np.array([0]))
    result.raw_groups = draws
    result.raw_parameters = raw
    return result


def posterior_parameters(
    model, estimates, parameter=None, num_steps=None, conditions=None, context=None, data_idx=None
):
    """Resolve paired posterior draws and deterministic curves at fixed conditions.

    Parameters
    ----------
    model : Model
        Model supplying parameter groups, fixed specifications, links, formulas,
        and deterministic transitions.
    estimates : mapping of str to np.ndarray
        Raw posterior draws with matching nonempty dataset and draw axes.
        Scalar trajectories have shape (B, S, T) or (B, S, T, 1); shared scalar
        coefficients have shape (B, S) or (B, S, 1). Shared vectors may use
        (B, S, K), but selected resolved quantities must be scalar per trial.
        Tiled shared arrays shaped (B, S, T, K) must be constant over time.
    parameter : str, sequence of str, or None, optional, default: None
        Scalar coefficients or formula targets to evaluate. If None, select
        available simulator parameters, excluding simulator-context variables.
    num_steps : int or None, optional, default: None
        Positive trajectory length. If None, infer it from posterior temporal
        arrays, then an available reference context array or DataFrame, otherwise
        use 200. An explicit length must match existing posterior trajectories;
        trajectories are never extrapolated.
    conditions : mapping, sequence of mappings, or None, optional, default: None
        Fixed predictor grid or explicit scalar assignments. If None, use the
        automatic choices described in `_make_conditions`.
    context : mapping or pandas.DataFrame or None, optional, default: None
        Reference predictor values for condition selection. If None, use the
        model's configured context source when a reference is required.
    data_idx : int, sequence of int, or None, optional, default: None
        Dataset indices to select. None includes all datasets; a sequence
        preserves the requested order, and a scalar retains the dataset axis.

    Returns
    -------
    draws : ConditionalDraws
        Resolved arrays shaped (B_selected, S, C, T), with original dataset
        indices and normalized `raw_parameters` retained for raw panels.
        `raw_groups` is not populated.

    Raises
    ------
    ValueError
        If posterior arrays, trajectory lengths, shared constancy, parameter
        selections, conditions, or resolved scalar values are invalid.
    KeyError
        If a required posterior parameter or inferred deterministic curve
        coefficient is missing.

    Notes
    -----
    Fixed values come from the model specifications. Missing deterministic
    trajectories are reconstructed from their paired curve-coefficient draws.
    Stochastic trajectories must be supplied. The function neither samples
    new parameters nor simulates observations, and does not alter `estimates`.
    """
    if not isinstance(estimates, Mapping) or not estimates:
        raise ValueError("estimates must be a non-empty mapping of posterior draws.")
    arrays = {name: np.asarray(value) for name, value in estimates.items()}
    first = next(iter(arrays.values()))
    if first.ndim < 2 or not first.shape[0] or not first.shape[1]:
        raise ValueError("Posterior arrays must start with non-empty (dataset, draw) axes.")
    datasets, draws = first.shape[:2]
    indices = normalize_data_indices(data_idx, datasets)
    temporal = set(model.local_keys) | set(model.deterministic_keys)
    lengths = {value.shape[2] for name, value in arrays.items() if name in temporal and value.ndim >= 3}
    if len(lengths) > 1:
        raise ValueError("Posterior trajectories have inconsistent trial counts.")
    if num_steps is None:
        if lengths:
            num_steps = next(iter(lengths))
        else:
            source = context if context is not None else model.context_simulator
            values = source.values() if isinstance(source, Mapping) else ()
            if isinstance(source, pd.DataFrame):
                num_steps = len(source)
            else:
                lengths_in_context = [np.asarray(v).shape[-1] for v in values if np.asarray(v).ndim in (1, 2)]
                num_steps = lengths_in_context[0] if lengths_in_context else 200
    _positive_integer(num_steps, "num_steps")
    if lengths and lengths != {num_steps}:
        raise ValueError("num_steps must match the posterior trajectory length; trajectories are not extrapolated.")

    raw = {}
    for name, array in arrays.items():
        if array.ndim not in (2, 3, 4) or array.shape[:2] != (datasets, draws):
            raise ValueError(f"Posterior parameter {name!r} has inconsistent dataset/draw axes or unsupported shape.")
        array = array[indices]
        if name in temporal:
            if array.ndim == 4 and array.shape[-1] == 1:
                array = array[..., 0]
            if array.shape != (len(indices), draws, num_steps):
                raise ValueError(f"Posterior trajectory {name!r} must be scalar-valued with {num_steps} steps.")
            raw[name] = array.reshape(-1, num_steps)
        else:
            if array.ndim == 4:
                if not np.all(array == array[:, :, :1, :]):
                    raise ValueError(f"Shared posterior parameter {name!r} changes across trials.")
                array = array[:, :, 0, :]
            if array.ndim == 3 and array.shape[-1] == 1:
                array = array[..., 0]
            raw[name] = array.reshape(len(indices) * draws, *array.shape[2:])

    # Fixed values come from specifications, never from new prior samples.
    for name, specification in model.prior.params.items():
        if np.isscalar(specification):
            raw.setdefault(name, specification)
        for key, spec in getattr(specification, "hyper_specs", {}).items():
            value, infer = specification._resolve(key, spec)
            if not infer:
                raw.setdefault(f"{name}_{key}", value)
        if isinstance(specification, DeterministicTransition) and name not in raw:
            curve_parameters = {}
            for key, spec in specification.hyper_specs.items():
                value, infer = specification._resolve(key, spec)
                full_name = f"{name}_{key}"
                if infer and full_name not in raw:
                    raise KeyError(f"Missing posterior curve coefficient {full_name!r}.")
                curve_parameters[key] = raw[full_name] if infer else value
                if not infer:
                    raw.setdefault(full_name, value)
            raw[name] = specification.sample_from_parameters(
                curve_parameters, batch_size=len(indices) * draws, num_steps=num_steps
            )

    names = _parameter_names(model, parameter)
    scenarios = _make_conditions(model, names, conditions, context, num_steps)
    result = _evaluate(model, raw, names, scenarios, len(indices), draws, num_steps, indices)
    result.raw_parameters = raw
    return result


def target_parameters(model, targets, names, draws):
    """Resolve ground truth using the posterior plot's conditions and dataset order.

    Parameters
    ----------
    model : Model
        Model defining raw coefficients, deterministic curves, links, and formulas.
    targets : mapping of str to np.ndarray or None
        Raw ground-truth coefficients with an original dataset axis and no
        posterior draw axis. Scalar trajectories use (B, T) or (B, T, 1), and
        shared coefficients use (B,) or (B, 1). Unrecognized keys are ignored.
        If None, no targets are evaluated.
    names : sequence of str
        Scalar parameters or formula targets to evaluate.
    draws : ConditionalDraws
        Nonempty posterior evaluation providing the exact fixed conditions,
        trajectory length, and original dataset indices to reuse.

    Returns
    -------
    truth : ConditionalDraws or None
        Resolved targets shaped (B_selected, 1, C, T), including raw coefficient
        metadata. Returns None when `targets` is None.

    Raises
    ------
    TypeError
        If supplied targets are not a named mapping of raw coefficients.

    Notes
    -----
    Targets are passed through `posterior_parameters` with a singleton draw axis,
    so links, formulas, fixed specifications, and deterministic reconstruction
    follow the same rules as the posterior evaluation.
    """
    if targets is None:
        return None
    if not isinstance(targets, Mapping):
        raise TypeError("Resolved plots require named targets containing the raw coefficients.")
    keys = set(model.prior.params) | set(model.hyper_keys) | set(model.deterministic_keys)
    samples = {name: np.asarray(value)[:, None, ...] for name, value in targets.items() if name in keys}
    return posterior_parameters(
        model,
        samples,
        names,
        num_steps=next(iter(draws.parameters.values())).shape[-1],
        conditions=draws.conditions,
        data_idx=draws.data_indices.tolist(),
    )


def posterior_invariant_samples(
    model,
    estimates,
    targets,
    variable_keys=None,
    conditions=None,
    context=None,
    num_steps=None,
    time_step=None,
    time_aggregation=None,
    scale="model",
):
    """Prepare invariant posterior samples without pooling time and draw uncertainty.

    Parameters
    ----------
    model : Model
        Model defining parameter groups, formulas, links, and predictor context.
    estimates : mapping of str to np.ndarray
        Raw posterior draws in the formats accepted by `posterior_parameters`.
        Vector parameters retain their component axis.
    targets : mapping of str to np.ndarray or None
        Raw ground-truth coefficients without a posterior draw axis, as accepted
        by `target_parameters`. None omits target evaluation.
    variable_keys : str, sequence of str, or None, optional, default: None
        Parameters to plot. If None, select hyperparameters, shared coefficients,
        and stationary simulator-facing formula targets. Explicit selections
        may include dynamic quantities with a time reduction specified below.
    conditions : mapping, sequence of mappings, or None, optional, default: None
        Fixed predictor grid or explicit scalar assignments. If None, use the
        automatic choices described in `_make_conditions`.
    context : mapping or pandas.DataFrame or None, optional, default: None
        Reference values for condition selection. If None, use the configured
        model context source when needed.
    num_steps : int or None, optional, default: None
        Reconstruction horizon or trajectory length, following the inference
        rules of `posterior_parameters`.
    time_step : int or None, optional, default: None
        Zero-based step to extract from dynamic quantities. Negative indices
        count from the end. Mutually exclusive with `time_aggregation`.
    time_aggregation : callable or None, optional, default: None
        Reduction called as time_aggregation(values, axis=-1) on resolved arrays
        shaped (B, S, C, T). Must return shape (B, S, C), preserving dataset,
        draw, and condition axes. Targets use the same reduction.
    scale : {"model", "both"}, optional, default: "model"
        Prepare implied quantities alone, or include paired raw coefficients
        under separate panel names. Identical views share a name; hyperparameters
        remain on their own scale. Public plot methods handle raw-only views.

    Returns
    -------
    samples : dict of str to np.ndarray
        Scalar and conditional samples shaped (B, S, C). Vector samples have
        shape (B, S, K), with K components. Neither time nor datasets are pooled
        into the posterior draw axis.
    target_samples : dict of str to np.ndarray or None
        Corresponding targets shaped (B, C), or (B, K) for vectors. None when
        no targets were supplied.
    labels : dict of str to list of str
        Labels for scalar parameters with multiple relevant fixed conditions.
    names : list of str
        Final sample keys in plotting order, including any raw/implied suffixes.

    Raises
    ------
    ValueError
        If no parameters are available, dynamic quantities lack a time selection,
        both reductions are specified, a step is out of range, an aggregation
        returns an invalid shape, or a shared vector changes across time.
    TypeError
        If `time_aggregation` is not callable or a required `time_step` is not
        a non-Boolean integer.
    KeyError
        If a required posterior parameter or vector target is unavailable.

    Notes
    -----
    Dynamic quantities are fully resolved, including links and formulas, before
    time reduction. Stationary quantities use their first step. Draws and targets
    follow the same condition selection, panel names, and reduction without
    posterior resampling. Vector components remain distinct from conditions.
    """
    names = invariant_parameter_names(model, variable_keys)
    if not names:
        raise ValueError("No time-invariant parameters found.")
    temporal = set(model.local_keys) | set(model.deterministic_keys)
    dynamic = {name for name in names if parameter_dependencies(model, [name]) & temporal}
    if dynamic and time_step is None and time_aggregation is None:
        raise ValueError("Time-varying parameters require an explicit time_step or time_aggregation for this plot.")
    if time_step is not None and time_aggregation is not None:
        raise ValueError("Choose time_step or time_aggregation, not both.")
    if time_aggregation is not None and not callable(time_aggregation):
        raise TypeError("time_aggregation must be callable.")
    scalar_names = [name for name in names if not _vector_parameter(model, name)]
    evaluated = (
        posterior_parameters(
            model,
            estimates,
            scalar_names,
            conditions=conditions,
            context=context,
            num_steps=num_steps,
        )
        if scalar_names
        else None
    )
    truth = target_parameters(model, targets, scalar_names, evaluated) if scalar_names and targets is not None else None
    if scale == "both" and scalar_names:
        raw_names = raw_parameter_names(model, names, kind="invariant", selected=variable_keys is not None)
        raw_names = [name for name in raw_names if not _vector_parameter(model, name)]
        evaluated, truth = parameter_views(model, evaluated, raw_names, truth)
        names = list(evaluated.parameters) + [name for name in names if _vector_parameter(model, name)]
        dynamic = {name for name in names if parameter_dependencies(model, [parameter_base_name(name)]) & temporal}
    result, target_result, labels = {}, {} if targets is not None else None, {}

    def reduce_time(name, values):
        """Select a step or reduce time while retaining dataset and draw axes.

        Parameters
        ----------
        name : str
            Parameter or panel name used to determine whether it is dynamic.
        values : np.ndarray
            Resolved values shaped (B, S, C, T).

        Returns
        -------
        reduced : np.ndarray
            Values shaped (B, S, C). Stationary parameters use their first
            step; dynamic parameters use the requested step or callable.

        Raises
        ------
        TypeError
            If a required time index is not a non-Boolean integer.
        ValueError
            If the index is out of range or the callable changes retained axes.
        """
        if name not in dynamic:
            return values[..., 0]
        if time_aggregation is not None:
            reduced = np.asarray(time_aggregation(values, axis=-1))
            if reduced.shape != values.shape[:-1]:
                raise ValueError("time_aggregation must return one value per dataset, draw, and condition.")
            return reduced
        if isinstance(time_step, bool) or not isinstance(time_step, Integral):
            raise TypeError("time_step must be an integer.")
        if not -values.shape[-1] <= time_step < values.shape[-1]:
            raise ValueError("time_step is outside the posterior trajectory.")
        return values[..., time_step]

    for name in names:
        if _vector_parameter(model, name):
            if name not in estimates:
                raise KeyError(f"Missing posterior parameter {name!r}.")
            values = np.asarray(estimates[name])
            if values.ndim == 4:
                if not np.allclose(values, values[:, :, :1, :]):
                    raise ValueError(f"Shared parameter {name!r} changes across time.")
                values = values[:, :, 0, :]
            raw_values = values
            if name in model.shared_keys:
                values = model._apply_links({name: values}, model.latent_link_function, [name])[name]
            raw_target, linked_target = None, None
            if targets is not None:
                raw_target = linked_target = np.asarray(targets[name])
                if name in model.shared_keys:
                    linked_target = model._apply_links({name: raw_target}, model.latent_link_function, [name])[name]
            vector_values, vector_targets = _vector_views(
                name, raw_values, values, scale=scale, raw_target=raw_target, implied_target=linked_target
            )
            result.update(vector_values)
            if targets is not None:
                target_result.update(vector_targets)
            continue
        indices, condition_names = condition_indices(model, evaluated, name)
        result[name] = reduce_time(name, evaluated.parameters[name])[:, :, indices]
        if truth is not None:
            target_result[name] = reduce_time(name, truth.parameters[name])[:, 0, indices]
        if len(indices) > 1:
            labels[name] = condition_names
    return result, target_result, labels, list(result)
