"""Shared data preparation helpers for plotting functions."""

from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import seaborn as sns
from matplotlib.axes import Axes


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
    bins: int = 40,
    density: bool = False,
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
    bins : int, optional, default: 40
        Number of histogram bins.
    density : bool, optional, default: False
        Whether a histogram-only plot should show density instead of counts.
        Histograms are always densities when combined with a KDE.
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

    values = np.asarray(values).reshape(-1)
    data = {"x": values} if orientation == "horizontal" else {"y": values}

    if dist_type == "hist":
        sns.histplot(
            **data,
            bins=bins,
            stat="density" if density else "count",
            color=color,
            alpha=1.0,
            label=label,
            ax=ax,
        )
    elif dist_type == "kde":
        sns.kdeplot(
            **data,
            color=color,
            fill=True,
            alpha=1.0,
            linewidth=2.0,
            label=label,
            ax=ax,
        )
    else:
        sns.histplot(
            **data,
            bins=bins,
            stat="density",
            color=color,
            alpha=1.0,
            ax=ax,
        )
        sns.kdeplot(
            **data,
            color=color,
            fill=False,
            alpha=1.0,
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
