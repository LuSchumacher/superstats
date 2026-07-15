"""Wrapper for missing at random data augmentation process"""

from .missing_process import MissingProcess
from superstats.defaults.augmentation_defaults import DEFAULT_P_MISSING_PRIOR
from superstats.prior.prior import Prior

import numpy as np


class RandomMissing(MissingProcess):
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
        Value written into masked entries. A scalar fills every data
        dimension; an array of shape ``(data_dim,)`` sets a
        per-dimension sentinel. Output dtype is promoted as needed
        (e.g. ``np.nan`` forces float; ``-1`` stays int on int data).
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
        controlled by the seed threaded through `GenerativeModel.sample`.
        """
        p = self.p_missing
        if isinstance(p, Prior):
            vals = np.asarray(p.sample(n), dtype=float).reshape(-1)
        else:
            vals = np.full(n, float(p))
        return np.clip(vals, 0.0, 1.0)

    def _fill(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        fill = np.asarray(self.missing_value)
        out = data.astype(np.result_type(data.dtype, fill.dtype), copy=True)
        out[mask] = np.broadcast_to(fill, out.shape)[mask]
        return out

    def apply(self, data: np.ndarray, rng: np.random.Generator | None = None) -> dict:
        rng = self._default_rng(rng)

        batch_size, num_steps = data.shape[0], data.shape[1]
        trailing = (1,) * (data.ndim - 2)

        if self.shared_across_batch:
            p = self._draw_p(1)[0]
            m = rng.random((num_steps,)) < p
            m = np.broadcast_to(m.reshape(1, num_steps), (batch_size, num_steps)).copy()
            p_used = np.full((batch_size, 1), p)
        else:
            p = self._draw_p(batch_size)
            m = rng.random((batch_size, num_steps)) < p[:, None]
            p_used = p.reshape(batch_size, 1)

        # full-shape bool mask, used only internally to fill `data`
        fill_mask = np.broadcast_to(m.reshape((batch_size, num_steps) + trailing), data.shape)

        # returned mask: 0/1 int, shape (batch_size, num_steps, 1)
        missing_mask = m.reshape(batch_size, num_steps, 1).astype(np.int64)

        return {
            "data": self._fill(data, fill_mask),
            "missing_mask": missing_mask,
            "p_missing": p_used,
            "missing_value": np.asarray(self.missing_value),
        }
