"""Wrapper for missing at random data augmentation process"""

from collections.abc import Mapping

from .missing import MissingProcess
from superstats.defaults.augmentation_defaults import DEFAULT_P_MISSING_PRIOR
from superstats.prior.prior import Prior

import numpy as np


class RandomMissingProcess(MissingProcess):
    """MCAR missingness with a per-dataset missing probability.

    Missingness is drawn per (batch, step): whenever a time step is
    selected as missing, all data dimensions at that step are set to
    `missing_value` (an entire observation is dropped, not individual
    features within it).

    Parameters
    ----------
    p_missing           : float, Prior, or None, default: None
        Probability that a time step is missing.
        - None (default): drawn from `DEFAULT_P_MISSING_PRIOR`, a
        Beta(2, 18) prior with mean 0.1.
        - float: fixed probability, shared across the whole batch.
        - Prior: sampled to obtain the probability. Sampled once for
        the whole batch if `shared_across_batch=True`, or once per
        dataset (default) otherwise.
        Prior draws (including the default) are clipped to [0, 1].
    missing_value       : float or np.ndarray, default: -1
        Value written into masked entries. A scalar fills every observed
        variable; a mapping sets a per-variable sentinel; an array of
        shape ``(num_variables,)`` sets sentinels in data-key order.
        Output dtype is promoted as needed (e.g. ``np.nan`` forces
        float; ``-1`` stays int on int data).
    shared_across_batch : bool, default: False
        If True, one probability and one mask are drawn and applied to
        every dataset in the batch. If False (default), each dataset
        gets its own probability draw and its own mask.
    """

    def __init__(
        self,
        p_missing: float | Prior | None = None,
        missing_value: float = -1,
        shared_across_batch: bool = False,
    ):
        self.p_missing = p_missing if p_missing is not None else DEFAULT_P_MISSING_PRIOR
        self.missing_value = missing_value
        self.shared_across_batch = shared_across_batch

    def _draw_p(self, n: int) -> np.ndarray:
        """Return `n` missing-probabilities in [0, 1].

        Note: `Prior.sample` draws from the global `np.random` state, not
        from the `rng` passed into `apply`, so draws from a `Prior` are not
        controlled by the seed threaded through `Model.sample`.
        """
        p = self.p_missing
        if isinstance(p, Prior):
            vals = p.sample(n)
        else:
            vals = np.full(n, p)
        return vals

    def _draw_mask(self, batch_size: int, num_steps: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Draw a batch x time missingness mask and the probabilities used."""
        if self.shared_across_batch:
            p = self._draw_p(1)[0]
            mask = rng.random(num_steps) < p
            mask = np.broadcast_to(mask[None, :], (batch_size, num_steps))
            p_used = np.full((batch_size, 1), p)
        else:
            p = self._draw_p(batch_size)
            mask = rng.random((batch_size, num_steps)) < p[:, None]
            p_used = p.reshape(batch_size, 1)
        return mask, p_used

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

    def apply(self, data: Mapping[str, np.ndarray], rng: np.random.Generator | None = None) -> dict:
        rng = self._default_rng(rng)

        data = {key: np.array(value, copy=True) for key, value in data.items()}

        keys = list(data)
        first = data[keys[0]]
        batch_size, num_steps = first.shape

        mask, p_used = self._draw_mask(batch_size, num_steps, rng)

        filled = {
            key: self._fill_array(value, mask, self._missing_value_for_key(key, i, len(keys)))
            for i, (key, value) in enumerate(data.items())
        }

        return filled | {
            "missing_mask": mask,
            "p_missing": p_used,
        }
