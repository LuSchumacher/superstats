"""Fixed categorical schemas shared by formulas and inference adapters."""

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np


def _level_kind(value: Any) -> str | None:
    if isinstance(value, (str, np.str_)):
        return "string"
    if isinstance(value, (bool, np.bool_)):
        return "boolean"
    if isinstance(value, Real) and np.isfinite(value):
        return "numeric"
    return None


@dataclass(frozen=True, init=False)
class Categorical:
    """Declare a nominal predictor with fixed levels and treatment coding.

    Parameters
    ----------
    levels
        Ordered sequence of at least two distinct string, finite numeric, or
        Boolean labels. All labels must belong to the same kind. Ordering is
        fixed across simulation batches and determines inference feature order.
    reference
        A declared level whose contribution in ``C(...)`` is zero. With an
        explicit intercept, the other coefficients are differences from it.

    Notes
    -----
    Formula terms require one effect for each non-reference level. Network
    inputs use full one-hot coding, including the reference level; numeric
    labels never imply an ordering or a linear effect.
    """

    levels: tuple[str | float | int | bool, ...]
    reference: str | float | int | bool

    def __init__(self, levels: Sequence, reference: str | float | int | bool):
        if isinstance(levels, (str, bytes)) or not isinstance(levels, (Sequence, np.ndarray)):
            raise TypeError("Categorical levels must be an ordered sequence of labels.")
        if isinstance(levels, np.ndarray) and levels.ndim != 1:
            raise ValueError("Categorical levels must be one-dimensional.")
        levels = tuple(value.item() if isinstance(value, np.generic) else value for value in levels)
        if len(levels) < 2:
            raise ValueError("Categorical requires at least two levels.")
        kinds = {_level_kind(value) for value in levels}
        if None in kinds or len(kinds) != 1:
            raise ValueError("Categorical levels must be all strings, all finite numbers, or all Booleans.")
        if len(set(levels)) != len(levels):
            raise ValueError("Categorical levels must be distinct.")
        if _level_kind(reference) not in kinds or reference not in levels:
            raise ValueError("Categorical reference must be one of the declared levels.")
        object.__setattr__(self, "levels", levels)
        object.__setattr__(self, "reference", levels[levels.index(reference)])

    @property
    def non_reference_levels(self) -> tuple:
        """Levels requiring effects in a treatment-coded formula term."""
        return tuple(level for level in self.levels if level != self.reference)

    def indices(self, values: Any, name: str = "predictor", missing_mask: Any = None) -> np.ndarray:
        """Validate labels and return fixed level indices (-1 for masked entries).

        Missing labels are accepted only at explicitly masked positions. In
        particular, an unknown observed label never becomes the reference.
        """
        # Object conversion preserves label types, including numeric sentinels
        # next to strings, rather than silently converting them to string labels.
        array = np.asarray(values, dtype=object)
        mask = np.zeros(array.shape, dtype=bool)
        if missing_mask is not None:
            supplied_mask = np.asarray(missing_mask)
            if supplied_mask.shape != array.shape:
                raise ValueError(f"Missing mask for categorical {name!r} must have shape {array.shape}.")
            if supplied_mask.dtype != bool and not np.all(np.isin(supplied_mask, (0, 1))):
                raise ValueError("Categorical missing_mask must contain only Boolean or 0/1 values.")
            mask = supplied_mask.astype(bool, copy=False)

        lookup = {level: index for index, level in enumerate(self.levels)}
        kind = _level_kind(self.levels[0])
        codes = np.full(array.shape, -1, dtype=np.intp)
        for index in np.flatnonzero(~mask):
            value = array.flat[index]
            code = lookup.get(value, -1) if _level_kind(value) == kind else -1
            if code == -1:
                raise ValueError(
                    f"Unknown or missing category {value!r} for {name!r}; expected one of {self.levels!r}."
                )
            codes.flat[index] = code
        return codes

    def encode(self, values: Any, name: str = "predictor", missing_mask: Any = None) -> np.ndarray:
        """Return float32 one-hot features, with zeros at masked positions."""
        codes = self.indices(values, name=name, missing_mask=missing_mask)
        encoded = np.eye(len(self.levels), dtype=np.float32)[np.maximum(codes, 0)]
        return np.where((codes >= 0)[..., None], encoded, 0.0)

    def get_config(self) -> dict:
        """Return a configuration containing only JSON-compatible values."""
        return {"levels": list(self.levels), "reference": self.reference}
