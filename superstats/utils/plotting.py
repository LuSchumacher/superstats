"""Shared data preparation helpers for plotting functions."""

from collections.abc import Mapping, Sequence

import numpy as np


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
