"""Wrapper for missing at random data augmentation process"""

from collections.abc import Mapping

from .missing import MissingProcess

import numpy as np


class RandomMissingProcess(MissingProcess):
    """MCAR missingness with a per-dataset missing probability.

    Specify ``p_missing`` in JointPrior. Model samples and links it once;
    it is always a nuisance parameter, never an inference target. Direct
    calls require an explicit final ``probability`` in [0, 1].

    Missingness is drawn per (batch, step): whenever a time step is
    selected as missing, all data dimensions at that step are set to
    `missing_value` (an entire observation is dropped, not individual
    features within it).

    Parameters
    ----------
    missing_value       : float or np.ndarray, default: -1
        Value written into masked entries. A scalar fills every observed
        variable; a mapping sets a per-variable sentinel; an array of
        shape ``(num_variables,)`` sets sentinels in data-key order.
        Output dtype is promoted as needed (e.g. ``np.nan`` forces
        float; ``-1`` stays int on int data).
    shared_across_batch : bool, default: False
        If True, one probability and one mask are drawn and applied to
        every dataset in the batch. If False (default), each dataset
        gets its own probability draw and its own mask. In shared mode,
        the first dataset's probability trajectory is used for the batch.
    """

    def __init__(
        self,
        missing_value: float = -1,
        shared_across_batch: bool = False,
    ):
        self.missing_value = missing_value
        self.shared_across_batch = shared_across_batch

    def __call__(self, data, rng=None, *, probability):
        return self.apply(data, rng=rng, probability=probability)

    def apply(self, data: Mapping[str, np.ndarray], rng=None, *, probability: float | np.ndarray) -> dict:
        """Apply missingness to a mapping of simulated data arrays."""
        rng = self._default_rng(rng)
        data = {key: np.array(value, copy=True) for key, value in data.items()}

        keys = list(data)
        first = data[keys[0]]
        batch_size, num_steps = first.shape
        mask, p_used = self._draw_mask(batch_size, num_steps, rng, probability)

        filled = {
            key: self._fill_array(value, mask, self._missing_value_for_key(key, index, len(keys)))
            for index, (key, value) in enumerate(data.items())
        }
        return filled | {"missing_mask": mask, "p_missing": p_used}

    def _draw_mask(self, batch_size, num_steps, rng, probability):
        """Use Model's final linked probability to draw an observation mask."""
        p = np.asarray(probability)
        if p.ndim == 3 and p.shape[-1] == 1:
            p = p[..., 0]
        if p.ndim == 0:
            p = np.full((batch_size, 1), p.item())
        elif p.shape == (batch_size,):
            p = p[:, None]
        if p.shape not in ((batch_size, 1), (batch_size, num_steps)):
            raise ValueError("probability must be scalar, per-dataset, or per-trial.")
        if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
            raise ValueError("p_missing must be between 0 and 1; configure Model.latent_link_functions.")
        if self.shared_across_batch:
            p = np.broadcast_to(p[:1], p.shape).copy()
            mask = np.broadcast_to((rng.random((1, num_steps)) < p[:1]), (batch_size, num_steps)).copy()
        else:
            mask = rng.random((batch_size, num_steps)) < p
        return mask, p

    @staticmethod
    def _fill_array(arr: np.ndarray, mask: np.ndarray, missing_value) -> np.ndarray:
        """Fill masked rows in one observed array, promoting dtype if needed."""
        try:
            arr[mask] = missing_value
        except (TypeError, ValueError, OverflowError):
            arr = arr.astype(np.result_type(arr.dtype, missing_value), copy=True)
            arr[mask] = missing_value
        return arr

    def _missing_value_for_key(self, key: str, index: int, num_keys: int):
        """Resolve scalar, per-key, or per-position missing values for mappings."""
        if isinstance(self.missing_value, Mapping):
            return self.missing_value[key]

        value = np.asarray(self.missing_value)
        if value.ndim == 0:
            return self.missing_value
        if value.shape != (num_keys,):
            raise ValueError(f"Array missing_value for mapping data must have shape ({num_keys},), got {value.shape}.")
        return value[index]
