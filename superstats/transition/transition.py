from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Union
import numpy as np

from superstats.prior import Prior
from superstats.defaults import (
    DEFAULT_HYPER_PRIORS,
    DEFAULT_BOUNDS,
    DEFAULT_INITIAL_PRIOR,
)

ParamSpec = Union[Prior, float, None]


class Transition(ABC):

    dtype = np.float32

    def __init__(
        self,
        bounds: Tuple[float, float] | np.ndarray | None = None,
        initial_prior: Prior | None = None,
    ):
        self._user_defined_bounds = bounds is not None
        self._user_defined_initial_prior = initial_prior is not None

        self.bounds = (
            np.asarray(bounds, dtype=self.dtype)
            if bounds is not None
            else np.asarray(DEFAULT_BOUNDS, dtype=self.dtype)
        )

        self.initial_prior = (
            initial_prior
            if initial_prior is not None
            else DEFAULT_INITIAL_PRIOR
        )
        self.hyper_specs: Dict[str, ParamSpec] = {}

    def _resolve(self, name: str, spec: ParamSpec) -> tuple[Prior | float, bool]:

        if spec is None:
            default = DEFAULT_HYPER_PRIORS.get(name)
            if default is None:
                raise KeyError(
                    f"No default hyperprior found for '{name}'"
                )
            return default, True

        if isinstance(spec, Prior):
            return spec, True

        if isinstance(spec, (float, int)):
            return float(spec), False

        raise TypeError(f"Invalid hyperparameter '{name}': {type(spec)}")

    def _sample(self, spec: Prior | float, batch_size: int) -> np.ndarray:
        if isinstance(spec, Prior):
            return spec.sample(batch_size).astype(self.dtype)

        return np.full(batch_size, spec, dtype=self.dtype)

    def _as_batch(self, x: np.ndarray | float, batch_size: int) -> np.ndarray:
        if np.ndim(x) == 0:
            return np.full(batch_size, x, dtype=self.dtype)
        return np.asarray(x, dtype=self.dtype)

    def _resolve_hyperparams(
        self,
        batch_size: int
    ) -> tuple[Dict[str, np.ndarray], Dict[str, float]]:

        hyper_params: Dict[str, np.ndarray] = {}
        fixed_params: Dict[str, float] = {}

        for name, spec in self.hyper_specs.items():
            value, infer = self._resolve(name, spec)

            if infer:
                hyper_params[name] = self._sample(value, batch_size)
            else:
                fixed_params[name] = value

        return hyper_params, fixed_params

    @abstractmethod
    def sample(
        self,
        batch_size: int,
        steps: int
    ) -> Dict[str, Any]:
        raise NotImplementedError
