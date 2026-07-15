"""Abstract base class for missing-data augmentation processes."""

from abc import ABC, abstractmethod
import numpy as np


class MissingProcess(ABC):
    """Introduces missingness into simulated data.

    Contract: ``(data, rng) -> {"data": filled, "missing_mask": mask}``.
    ``mask`` is a boolean array of ``data.shape`` (True = missing) and
    ``filled`` is ``data`` with masked entries set to the process's
    ``missing_value``. Instances are callable, so a MissingProcess, a
    subclass, or a bare function with this signature are interchangeable.
    """

    @staticmethod
    def _default_rng(rng: np.random.Generator | None) -> np.random.Generator:
        """Single source of truth for the "no rng supplied" fallback.

        Both `apply` (called directly) and `__call__` (called via the
        instance) route through this, so there is one definition of what
        "no rng" means, even though both are safe to call without one.
        """
        return np.random.default_rng() if rng is None else rng

    @abstractmethod
    def apply(self, data: np.ndarray, rng: np.random.Generator | None = None) -> dict:
        """Apply the missingness process.

        Parameters
        ----------
        data : np.ndarray
            Simulated data to corrupt with missingness.
        rng  : np.random.Generator or None, optional, default: None
            Random generator to use. If None, a fresh, unseeded generator
            is created via `_default_rng`, so calling `apply` directly is
            safe but not reproducible unless a seeded `rng` is supplied.

        Returns
        -------
        result : dict with keys "data" and "missing_mask"
        """
        ...

    def __call__(self, data: np.ndarray, rng: np.random.Generator | None = None) -> dict:
        return self.apply(data, self._default_rng(rng))
