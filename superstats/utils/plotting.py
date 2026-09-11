"""Shared data preparation and rendering helpers for plotting functions."""

from collections.abc import Callable, Mapping, Sequence
from typing import Literal

import numpy as np
import seaborn as sns
from matplotlib.axes import Axes

from superstats.defaults import (
    DIST_ALPHA,
    OVERLAY_DIST_ALPHA,
    UNCERTAINTY_BAND_LABELS,
    UNCERTAINTY_INTERVAL_LABELS,
)


def prepare_time_invariant_data(
    estimates: Mapping[str, np.ndarray] | np.ndarray,
    targets: Mapping[str, np.ndarray] | np.ndarray | None,
    variable_keys: Sequence[str] | None,
    variable_names: Sequence[str] | None,
    mixture_names: Mapping[str, Sequence[str]] | None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None, list[str], dict[str, list[str]]]:
    """Normalize time-invariant posterior inputs while preserving components."""
    mixture_names = mixture_names or {}
    is_mapping = isinstance(estimates, Mapping)
    if targets is not None and is_mapping != isinstance(targets, Mapping):
        raise ValueError("estimates and targets must both be mappings or both be arrays.")

    if is_mapping:
        keys = list(variable_keys) if variable_keys is not None else list(estimates.keys())
        if not keys:
            raise ValueError("No variables found to plot.")
        missing = [key for key in keys if key not in estimates]
        if targets is not None:
            missing.extend(key for key in keys if key not in targets and key not in missing)
        if missing:
            raise ValueError(f"variable_keys not found in estimates and targets: {missing}")
        names = list(variable_names) if variable_names is not None else keys
        if len(names) != len(keys):
            raise ValueError(f"variable_names has {len(names)} entries but there are {len(keys)} variables.")
        local_estimates = {name: np.asarray(estimates[key]) for key, name in zip(keys, names)}
        local_targets = (
            {name: np.asarray(targets[key]) for key, name in zip(keys, names)} if targets is not None else None
        )
        local_mixture_names = {}
        for key, name in zip(keys, names):
            base_name = key.split("_mixture_weights")[0]
            if base_name in mixture_names:
                local_mixture_names[name] = list(mixture_names[base_name])
    else:
        estimates_arr = np.asarray(estimates)
        if estimates_arr.ndim != 4:
            raise ValueError("Array estimates must have shape (num_datasets, num_post_samples, num_steps, num_params).")
        num_params = estimates_arr.shape[-1]
        names = list(variable_names) if variable_names is not None else [f"param_{p}" for p in range(num_params)]
        if not names:
            raise ValueError("No variables found to plot.")
        if len(names) != num_params:
            raise ValueError(f"variable_names has {len(names)} entries but there are {num_params} variables.")
        local_estimates = {name: estimates_arr[..., p : p + 1] for p, name in enumerate(names)}
        if targets is None:
            local_targets = None
        else:
            targets_arr = np.asarray(targets)
            if targets_arr.ndim not in (2, 3) or targets_arr.shape[-1] != num_params:
                raise ValueError(
                    "Array targets must have shape (num_datasets, num_params) or (num_datasets, num_steps, num_params)."
                )
            local_targets = {name: targets_arr[..., p : p + 1] for p, name in enumerate(names)}
        local_mixture_names = {}

    num_datasets = None
    sample_shape = None
    for name, values in local_estimates.items():
        if values.ndim != 4:
            raise ValueError(
                f"Estimates for '{name}' must have shape (num_datasets, num_post_samples, num_steps, num_components)."
            )
        if num_datasets is None:
            num_datasets = values.shape[0]
            sample_shape = values.shape[1:3]
        elif values.shape[0] != num_datasets:
            raise ValueError("All estimate variables must have the same number of datasets.")
        elif values.shape[1:3] != sample_shape:
            raise ValueError("All estimate variables must have the same sample and step dimensions.")
        if name in local_mixture_names and len(local_mixture_names[name]) != values.shape[-1]:
            raise ValueError(
                f"mixture_names for '{name}' has {len(local_mixture_names[name])} entries "
                f"but the variable has {values.shape[-1]} components."
            )
        if local_targets is not None:
            local_targets[name] = normalize_time_invariant_target(
                name,
                local_targets[name],
                values.shape[0],
                values.shape[-1],
            )

    return local_estimates, local_targets, names, local_mixture_names


def normalize_time_invariant_target(
    name: str,
    values: np.ndarray,
    num_datasets: int,
    num_components: int,
) -> np.ndarray:
    """Collapse a step-expanded invariant target to dataset by component."""
    target = np.asarray(values)
    if target.shape[0] != num_datasets:
        raise ValueError(f"Target '{name}' has {target.shape[0]} datasets, expected {num_datasets}.")
    if target.ndim == 1 and num_components == 1:
        return target[:, None]
    if target.ndim == 2 and target.shape[1] == num_components:
        return target
    if target.ndim == 2 and num_components == 1:
        target = target[..., None]
    elif target.ndim != 3 or target.shape[-1] != num_components:
        raise ValueError(
            f"Targets for '{name}' must have shape ({num_datasets}, {num_components}) or "
            f"({num_datasets}, num_steps, {num_components}), got {target.shape}."
        )
    if not np.allclose(target, target[:, :1, :], equal_nan=True):
        raise ValueError(f"Target '{name}' varies across steps but is declared time-invariant.")
    return target[:, 0, :]


def flatten_time_invariant_parameters(
    estimates: dict[str, np.ndarray],
    targets: dict[str, np.ndarray] | None,
    names: Sequence[str],
    mixture_names: Mapping[str, Sequence[str]],
) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    """Flatten posterior sample/step axes and resample the draw count."""
    sample_columns = []
    target_columns = []
    parameter_names = []
    for name in names:
        values = estimates[name]
        num_components = values.shape[-1]
        component_names = mixture_names.get(name)
        for component in range(num_components):
            sample_columns.append(values[..., component].reshape(values.shape[0], -1))
            if targets is not None:
                target_columns.append(targets[name][:, component])
            if num_components == 1:
                parameter_names.append(name)
            elif component_names is None:
                parameter_names.append(f"{name}[{component}]")
            else:
                display_name = name.removesuffix("_mixture_weights")
                parameter_names.append(f"{display_name}: {component_names[component]}")

    flattened_samples = np.stack(sample_columns, axis=-1)
    num_datasets = flattened_samples.shape[0]
    num_samples = estimates[names[0]].shape[1]
    generator = np.random.default_rng()
    flat_indices = np.stack(
        [generator.choice(flattened_samples.shape[1], size=num_samples, replace=False) for _ in range(num_datasets)]
    )
    samples = np.take_along_axis(flattened_samples, flat_indices[..., None], axis=1)
    target_values = np.stack(target_columns, axis=-1) if targets is not None else None
    return samples, target_values, parameter_names


def select_data_variable(
    data: Mapping[str, np.ndarray],
    data_dim: int | str,
) -> np.ndarray:
    """Resolve named data to one ``(batch_size, num_steps)`` variable."""
    if not isinstance(data, Mapping):
        raise TypeError(f"data must be a mapping of named arrays, got {type(data)}.")

    keys = list(data)
    if isinstance(data_dim, int):
        try:
            key = keys[data_dim]
        except IndexError as exc:
            raise ValueError(f"data_dim index {data_dim} is out of range for data keys {keys!r}.") from exc
    else:
        key = data_dim
        if key not in data:
            raise KeyError(f"data key {key!r} not found. Available keys: {keys!r}.")

    values = np.asarray(data[key])
    if values.ndim != 2:
        raise ValueError(f"Data variable {key!r} must have shape (batch_size, steps), got {values.shape}.")
    return values


def resolve_dist_alpha(
    dist_alpha: float | None,
    num_distributions: int,
) -> float:
    """Resolve automatic distribution opacity from the number of overlays."""
    if num_distributions < 1:
        raise ValueError("num_distributions must be at least 1.")
    if dist_alpha is None:
        return DIST_ALPHA if num_distributions == 1 else OVERLAY_DIST_ALPHA
    if not 0 <= dist_alpha <= 1:
        raise ValueError("dist_alpha must be between 0 and 1.")
    return dist_alpha


def get_uncertainty_band_label(
    uncertainty_fun: str | Callable,
) -> str:
    """Return the shared legend label for an uncertainty-band method."""
    return UNCERTAINTY_BAND_LABELS[uncertainty_fun] if isinstance(uncertainty_fun, str) else "Uncertainty"


def get_uncertainty_interval_labels(
    uncertainty_fun: str | Callable,
) -> tuple[str, str | None]:
    """Return separate outer and inner labels for a forest-plot interval."""
    if callable(uncertainty_fun):
        return "Uncertainty", None
    return UNCERTAINTY_INTERVAL_LABELS[uncertainty_fun]


def smooth_trajectories(
    arr: np.ndarray,
    smoothing: Literal["sma", "ema"] | None,
    smoothing_window: int = 5,
) -> np.ndarray:
    """Apply causal SMA or EMA smoothing along the last array axis."""
    if smoothing is None:
        return arr
    if smoothing not in {"sma", "ema"}:
        raise ValueError(f"smoothing must be None, 'sma', or 'ema', got {smoothing!r}.")
    if smoothing_window < 1:
        raise ValueError("smoothing_window must be at least 1.")

    smoothed = arr.copy()
    if smoothing == "sma":
        for i in range(arr.shape[-1]):
            start = max(0, i - smoothing_window + 1)
            smoothed[..., i] = arr[..., start : i + 1].mean(axis=-1)
    else:
        alpha = 2.0 / (smoothing_window + 1)
        for i in range(1, arr.shape[-1]):
            smoothed[..., i] = alpha * arr[..., i] + (1 - alpha) * smoothed[..., i - 1]
    return smoothed


def _compute_hdi(
    trajectories: np.ndarray,
    probability: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a per-step highest-density interval."""
    num_steps = trajectories.shape[-1]
    lower, upper = np.empty(num_steps), np.empty(num_steps)
    for i in range(num_steps):
        values = np.sort(trajectories[:, i])
        num_values = len(values)
        if num_values == 1:
            lower[i] = upper[i] = values[0]
            continue
        window = max(1, min(num_values - 1, int(np.floor(probability * num_values))))
        widths = values[window:] - values[: num_values - window]
        index = np.argmin(widths)
        lower[i], upper[i] = values[index], values[index + window]
    return lower, upper


def compute_uncertainty_band(
    trajectories: np.ndarray,
    uncertainty_fun: Literal["std", "ci", "mad", "hdi"] | Callable,
    center: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute an uncertainty interval across trajectory axis 0."""
    if callable(uncertainty_fun):
        result = uncertainty_fun(trajectories)
        if len(result) != 2:
            raise ValueError("Custom uncertainty_fun must return (lower, upper).")
        return np.asarray(result[0]), np.asarray(result[1])
    if uncertainty_fun == "std":
        sd = trajectories.std(axis=0)
        return center - sd, center + sd
    if uncertainty_fun == "ci":
        return (
            np.percentile(trajectories, 2.5, axis=0),
            np.percentile(trajectories, 97.5, axis=0),
        )
    if uncertainty_fun == "mad":
        median = np.median(trajectories, axis=0)
        mad = np.median(np.abs(trajectories - median), axis=0)
        scaled_mad = 1.48 * mad
        return center - scaled_mad, center + scaled_mad
    if uncertainty_fun == "hdi":
        return _compute_hdi(trajectories, probability=0.95)
    raise ValueError(f"uncertainty_fun must be 'std', 'ci', 'mad', 'hdi', or callable. Got {uncertainty_fun!r}.")


def compute_uncertainty_bands(
    trajectories: np.ndarray,
    uncertainty_fun: Literal["std", "ci", "mad", "hdi"] | Callable,
    center: np.ndarray,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray] | None]:
    """Compute outer and inner uncertainty intervals across trajectory axis 0.

    Named uncertainty methods return two nested intervals: ±1 and ±0.5
    standard deviations, central 95% and 65% intervals, ±1.48 and
    ±0.74 median absolute deviations, or 95% and 65% highest-density
    intervals. A custom callable defines one interval, so its inner interval
    is ``None``.
    """
    outer = compute_uncertainty_band(trajectories, uncertainty_fun, center)
    if callable(uncertainty_fun):
        return outer, None

    if uncertainty_fun == "std":
        half_sd = 0.5 * trajectories.std(axis=0)
        inner = (center - half_sd, center + half_sd)
    elif uncertainty_fun == "ci":
        inner = (
            np.percentile(trajectories, 17.5, axis=0),
            np.percentile(trajectories, 82.5, axis=0),
        )
    elif uncertainty_fun == "mad":
        median = np.median(trajectories, axis=0)
        mad = np.median(np.abs(trajectories - median), axis=0)
        half_scaled_mad = 0.74 * mad
        inner = (center - half_scaled_mad, center + half_scaled_mad)
    else:
        inner = _compute_hdi(trajectories, probability=0.65)
    return outer, inner


def plot_uncertainty_band(
    ax: Axes,
    steps: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    color: str,
    alpha: float,
    zorder: float = 1,
) -> bool:
    """Draw a non-zero-width uncertainty band and report its visibility."""
    finite = np.isfinite(lower) & np.isfinite(upper)
    visible = np.any(finite & ~np.isclose(lower, upper))
    if visible:
        ax.fill_between(
            steps,
            lower,
            upper,
            color=color,
            alpha=alpha,
            edgecolor="none",
            zorder=zorder,
        )
    return bool(visible)


def plot_uncertainty_bands(
    ax: Axes,
    steps: np.ndarray,
    outer: tuple[np.ndarray, np.ndarray],
    inner: tuple[np.ndarray, np.ndarray] | None,
    color: str,
    alpha: float,
    zorder: float = 1,
) -> bool:
    """Draw a bright outer ribbon and, when available, a darker inner ribbon."""
    visible = plot_uncertainty_band(
        ax,
        steps,
        outer[0],
        outer[1],
        color,
        alpha=alpha * 0.5,
        zorder=zorder,
    )
    if inner is not None:
        visible |= plot_uncertainty_band(
            ax,
            steps,
            inner[0],
            inner[1],
            color,
            alpha=alpha,
            zorder=zorder + 0.1,
        )
    return visible


def get_default_num_cols(
    num_panels: int,
) -> int:
    """Return the compact default number of plot columns."""

    num_cols = {
        1: 1,
        2: 2,
        3: 3,
        4: 2,
        5: 3,
        6: 3,
        7: 4,
        8: 4,
        9: 3,
        10: 4,
    }
    return num_cols.get(num_panels, 4)


def get_layout(
    num_rows: int,
    num_cols: int,
    figsize: tuple[float, float] | None = None,
    col_width: float = 1.0,
    row_height: float = 1.0,
    legend_space: float = 1.6,
    legend_offset: float = 0.1,
) -> tuple[tuple[float, float], float, float]:
    """Return figure size, bottom margin, and legend anchor.

    The layout is calculated in inches before being converted to figure
    coordinates. This keeps the legend spacing constant as rows are added.
    """
    if figsize is None:
        figsize = (
            col_width * num_cols,
            row_height * num_rows + legend_space,
        )

    legend_bottom = legend_space / figsize[1]
    legend_y = legend_offset / figsize[1]

    return figsize, legend_bottom, legend_y


def plot_dist(
    values: np.ndarray,
    ax: Axes,
    dist_type: Literal["hist", "kde", "both"],
    color: str,
    orientation: Literal["horizontal", "vertical"] = "horizontal",
    num_bins: int | None = None,
    alpha: float = DIST_ALPHA,
    label: str | None = None,
    hide_axis: bool = False,
) -> None:
    """Plot a histogram, KDE, or both.

    Parameters
    ----------
    values : np.ndarray
        Values to plot.
    ax : matplotlib.axes.Axes
        Matplotlib axis on which to draw the distribution.
    dist_type : {"hist", "kde", "both"}
        Plot type.
    color : str
        Plot color.
    orientation : {"horizontal", "vertical"}, optional, default: "horizontal"
        Put values on the x-axis (horizontal) or y-axis (vertical).
    num_bins : int or None, optional, default: None
        Number of histogram bins. If None, Seaborn selects the bins.
    alpha : float, optional, default: DIST_ALPHA
        Opacity of the histogram or KDE.
    label : str or None, optional, default: None
        Legend label for the distribution.
    hide_axis : bool, optional, default: False
        Whether to hide the axis after plotting.

    Raises
    ------
    ValueError
        If ``dist_type`` or ``orientation`` is invalid.
    """
    if dist_type not in {"hist", "kde", "both"}:
        raise ValueError("dist_type must be one of 'hist', 'kde', or 'both'.")
    if orientation not in {"horizontal", "vertical"}:
        raise ValueError("orientation must be 'horizontal' or 'vertical'.")
    if num_bins is not None and num_bins < 1:
        raise ValueError("num_bins must be at least 1.")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1.")

    values = np.asarray(values).reshape(-1)
    data = {"x": values} if orientation == "horizontal" else {"y": values}
    hist_kwargs = {} if num_bins is None else {"bins": num_bins}

    if dist_type == "hist":
        sns.histplot(
            **data,
            **hist_kwargs,
            stat="density",
            color=color,
            alpha=alpha,
            label=label,
            ax=ax,
        )
    elif dist_type == "kde":
        sns.kdeplot(
            **data,
            color=color,
            fill=True,
            alpha=alpha,
            linewidth=0,
            label=label,
            ax=ax,
        )
    else:
        sns.histplot(
            **data,
            **hist_kwargs,
            stat="density",
            color=color,
            alpha=alpha,
            ax=ax,
        )
        sns.kdeplot(
            **data,
            color=color,
            fill=False,
            alpha=alpha,
            linewidth=2.0,
            label=label,
            ax=ax,
        )

    if hide_axis:
        ax.set_axis_off()


def prepare_plot_data(
    estimates: Mapping[str, np.ndarray] | np.ndarray,
    targets: Mapping[str, np.ndarray] | np.ndarray,
    variable_keys: Sequence[str] | None = None,
    variable_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Resolve dict-or-array estimates/targets into stacked arrays plus display names.

    For dict input, per-key arrays are stacked along a new last axis in
    the order given by `variable_keys` (or all keys, by default). For
    array input, `estimates`/`targets` are used as-is and `variable_keys`
    is ignored.

    Parameters
    ----------
    estimates      : Mapping[str, np.ndarray] or np.ndarray
        If a dict, per-key arrays sharing the same leading shape, to
        be stacked into a new last axis. If an array, used directly
        (already includes the params axis). Must be the same kind
        (dict or array) as `targets`.
    targets        : Mapping[str, np.ndarray] or np.ndarray
        Same convention as `estimates`, one fewer axis than
        `estimates` in the array case (no sample axis). Must be the
        same kind (dict or array) as `estimates`.
    variable_keys  : sequence of str or None, optional, default: None
        Which keys to select from `estimates`/`targets` when they are
        dicts, and in what order. By default, all keys, in dict
        insertion order. Ignored if `estimates`/`targets` are arrays.
    variable_names : sequence of str or None, optional, default: None
        Display names for the columns, in the same order as the
        selected variables. Defaults to `variable_keys` (if dicts) or
        `param_0`, `param_1`, ... (if arrays).

    Returns
    -------
    estimates_arr, targets_arr, names : tuple
        Stacked arrays of shape (..., num_params) and a list of
        display names, one per column.

    Raises
    ------
    ValueError
        If `estimates` and `targets` are not both dicts or both
        arrays, if `variable_keys` references a key missing from
        either dict, or if `variable_names` doesn't match the number
        of selected/resolved variables.
    """
    is_mapping = isinstance(estimates, Mapping)
    if is_mapping != isinstance(targets, Mapping):
        raise ValueError("estimates and targets must both be dicts or both be arrays.")

    if is_mapping:
        keys = list(variable_keys) if variable_keys is not None else list(estimates.keys())
        missing = [k for k in keys if k not in estimates or k not in targets]
        if missing:
            raise ValueError(f"variable_keys not found in both estimates and targets: {missing}")

        estimates_arr = np.stack([estimates[k] for k in keys], axis=-1)
        targets_arr = np.stack([targets[k] for k in keys], axis=-1)
        names = list(variable_names) if variable_names is not None else keys
    else:
        num_params = estimates.shape[-1]
        estimates_arr = estimates
        targets_arr = targets
        names = list(variable_names) if variable_names is not None else [f"param_{p}" for p in range(num_params)]

    if len(names) != estimates_arr.shape[-1]:
        raise ValueError(f"variable_names has {len(names)} entries but there are {estimates_arr.shape[-1]} variables.")

    return estimates_arr, targets_arr, names
