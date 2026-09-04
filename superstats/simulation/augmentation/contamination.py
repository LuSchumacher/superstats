"""Abstract base class for contamination-data augmentation processes."""

from abc import ABC, abstractmethod
import numpy as np


class ContaminationProcess(ABC):
    """Introduces contamination into simulated data.

    Contract: ``(data, rng) -> {"data": contaminated}``.
    ``mask`` is a boolean array of ``data.shape`` (True = contaminated) and
    ``contaminated`` is ``data`` with masked entries replaced by draws from
    the process's contaminant distribution (e.g. guesses, lapses, outliers).
    Instances are callable, so a ContaminationProcess, a subclass, or a bare
    function with this signature are interchangeable.
    """

    @abstractmethod
    def apply(self, data: np.ndarray, rng: np.random.Generator | None = None) -> dict:
        """Apply the contamination process.

        Parameters
        ----------
        data : np.ndarray
            Simulated data to corrupt with contamination.
        rng  : np.random.Generator or None, optional, default: None
            Random generator to use. If None, a fresh, unseeded generator
            is created via `_default_rng`, so calling `apply` directly is
            safe but not reproducible unless a seeded `rng` is supplied.

        Returns
        -------
        result : dict with keys "data" and "contamination_mask"
        """
        ...

    def __call__(self, data: np.ndarray, rng: np.random.Generator | None = None) -> dict:
        return self.apply(data, self._default_rng(rng))

    @staticmethod
    def _default_rng(rng: np.random.Generator | None) -> np.random.Generator:
        """Single source of truth for the "no rng supplied" fallback.

        Both `apply` (called directly) and `__call__` (called via the
        instance) route through this, so there is one definition of what
        "no rng" means, even though both are safe to call without one.
        """
        return np.random.default_rng() if rng is None else rng
