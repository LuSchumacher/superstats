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

        self.hyper_specs: Dict[str, ParamSpec] = {}

    # -------------------------
    # resolution
    # -------------------------

    def _resolve(self, name: str, spec: ParamSpec) -> tuple[Prior | float, bool]:
        # Resolve parameter into (value, infer_flag).

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

    def _resolve_hyperparams(
        self,
        batch_size: int
    ) -> tuple[Dict[str, np.ndarray], Dict[str, float]]:
        # Resolve hyperparameters into sampled and fixed components.

        hyper_params: Dict[str, np.ndarray] = {}
        fixed_params: Dict[str, float] = {}

        for name, spec in self.hyper_specs.items():
            value, infer = self._resolve(name, spec)
            if infer:
                hyper_params[name] = self._sample(value, batch_size)
            else:
                fixed_params[name] = value

        return hyper_params, fixed_params

    def _sample(self, value: Prior | float, batch_size: int) -> np.ndarray:
        if isinstance(value, Prior):
            return value.sample(batch_size).astype(self.dtype)

        return np.full(batch_size, value, dtype=self.dtype)

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