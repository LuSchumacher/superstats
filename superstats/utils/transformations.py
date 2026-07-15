"""Transformation functions."""

import numpy as np
import pandas as pd
from numba import njit


@njit
def scaled_sigmoid(
    x: float | np.ndarray, lower_bound: float | np.ndarray, upper_bound: float | np.ndarray
) -> float | np.ndarray:
    """Apply a sigmoid transformation and rescale to a bounded interval.

    This function maps input values to a specified range using a scaled
    sigmoid. The transformation is:
    lower_bound + (upper_bound - lower_bound) / (1 + exp(-x))

    Parameters
    ----------
    x           : float or np.ndarray
        Input value(s) to transform.
    lower_bound : float or np.ndarray
        Lower bound of the output range.
    upper_bound : float or np.ndarray
        Upper bound of the output range.

    Returns
    -------
    y : float or np.ndarray - transformed value(s) in
        [lower_bound, upper_bound]
    """
    return lower_bound + (upper_bound - lower_bound) / (1.0 + np.exp(-x))


def df_to_array(df: pd.DataFrame, id_col: str, data_cols: tuple[str, ...]) -> np.ndarray:
    """Reshape a long-format DataFrame into (num_datasets, num_steps, data_dims).

    Each unique value in `id_col` becomes one dataset (one row of the first
    axis). Row order within a dataset is preserved as given in `df` (sort
    beforehand if you need a specific trial order, e.g. by a trial-index
    column). Datasets with fewer rows than the longest dataset are padded
    with NaN at the end.

    Parameters
    ----------
    df        : pd.DataFrame
        Long-format data with one row per (dataset, step).
    id_col    : str, default: "id"
        Column identifying which dataset a row belongs to.
    data_cols : tuple of str, default: ("rt", "correct")
        Columns to stack into the trailing `data_dims` axis, in order.

    Returns
    -------
    data : np.ndarray of shape (num_datasets, num_steps, len(data_cols))
        `num_steps` is the length of the longest dataset; shorter
        datasets are NaN-padded at the end.
    """
    groups = [g[list(data_cols)].to_numpy(dtype=float) for _, g in df.groupby(id_col, sort=False)]

    num_datasets = len(groups)
    num_steps = max(g.shape[0] for g in groups)
    data_dims = len(data_cols)

    data = np.full((num_datasets, num_steps, data_dims), np.nan, dtype=np.float32)
    for i, g in enumerate(groups):
        data[i, : g.shape[0], :] = g

    return data
