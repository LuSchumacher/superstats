from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Union
import numpy as np

from superstats.prior import Prior
from superstats.defaults import DEFAULT_GLOBAL_PRIORS


ParamSpec = Union[Prior, float, None]


class Transition(ABC):
    """
    Base class for stochastic parameter transition processes.

    Core responsibilities:
    - define parameter resolution rules
    - sample global parameters
    - expose inference mask
    - provide shared configuration (bounds, initial prior)
    """

    def __init__(
        self,
        bounds: Tuple[float, float] | np.ndarray,
        initial_prior: Prior | None = None,
        dtype: np.dtype = np.float32,
    ):
        self.dtype = dtype
        self.bounds = np.asarray(bounds, dtype=dtype)

        self.initial_prior = (
            initial_prior
            if initial_prior is not None
            else Prior("normal", loc=0.0, scale=1.0)
        )

        self.global_params: Dict[str, ParamSpec] = {}

    # -------------------------
    # resolution
    # -------------------------

    def _resolve(self, name: str, spec: ParamSpec) -> tuple[Prior | float, bool]:
        """
        Resolve parameter into (value, infer_flag).
        """

        # None -> default prior
        if spec is None:
            default = DEFAULT_GLOBAL_PRIORS.get(f"{name}_prior")
            if default is None:
                raise KeyError(
                    f"No default prior found for '{name}' "
                    f"(expected '{name}_prior')."
                )
            return default, True

        # Prior -> infer
        if isinstance(spec, Prior):
            return spec, True

        # float -> fixed
        return spec, False

    # -------------------------
    # sampling
    # -------------------------

    def _sample(self, value: Prior | float, batch_size: int) -> np.ndarray:
        if isinstance(value, Prior):
            return value.sample(batch_size).astype(self.dtype)

        return np.full(batch_size, value, dtype=self.dtype)

    def sample_global_params(
        self,
        batch_size: int
    ) -> tuple[Dict[str, np.ndarray], Dict[str, bool]]:
        """
        Sample global parameters + return inference mask.
        """

        values: Dict[str, np.ndarray] = {}
        infer_flags: Dict[str, bool] = {}

        for name, spec in self.global_params.items():
            value, infer = self._resolve(name, spec)
            values[name] = self._sample(value, batch_size)
            infer_flags[name] = infer

        return values, infer_flags

    # -------------------------
    # interface
    # -------------------------

    @abstractmethod
    def sample(
        self,
        batch_size: int,
        steps: int
    ) -> Dict[str, Any]:
        raise NotImplementedError