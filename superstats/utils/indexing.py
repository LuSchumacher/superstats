"""Shared helpers for selecting datasets by index."""

from collections.abc import Sequence
from numbers import Integral

import numpy as np


def format_dataset_label(index: int) -> str:
    """Return a one-based display label for a zero-based dataset index."""
    return f"Dataset {index + 1}"


def normalize_data_indices(
    data_idx: int | Sequence[int] | None,
    num_datasets: int,
) -> np.ndarray:
    """Return validated dataset indices while preserving selection order."""
    if data_idx is None:
        return np.arange(num_datasets)

    if isinstance(data_idx, Integral) and not isinstance(data_idx, bool):
        raw_indices = [int(data_idx)]
    elif isinstance(data_idx, Sequence) and not isinstance(data_idx, (str, bytes)):
        raw_indices = list(data_idx)
        if not raw_indices:
            raise ValueError("data_idx must contain at least one dataset index.")
        if any(not isinstance(index, Integral) or isinstance(index, bool) for index in raw_indices):
            raise TypeError("data_idx must be an int or a sequence of ints.")
    else:
        raise TypeError("data_idx must be None, an int, or a sequence of ints.")

    indices = np.asarray(raw_indices, dtype=int)
    indices = np.where(indices < 0, indices + num_datasets, indices)
    invalid = indices[(indices < 0) | (indices >= num_datasets)]
    if invalid.size:
        raise ValueError(f"data_idx contains out-of-range index {invalid[0]} for {num_datasets} datasets.")
    return indices
