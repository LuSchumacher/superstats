"""Base interface and shared helpers for deterministic transitions."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Dict, Any, Union
import numpy as np

from superstats.defaults import (
    DEFAULT_DETERMINISTIC_HYPER_PRIORS,
    DEFAULT_INITIAL_PRIOR,
    DEFAULT_BOUNDS,
)
from superstats.prior.prior import Prior

ParamSpec = Union[Prior, float, int, None]


class DeterministicTransition(ABC):
    """Base class for deterministic transition models.

    Subclasses implement deterministic dynamics for a single scalar latent
    parameter. Subclasses must implement `sample` to generate complete
    trajectories from resolved parameter values.

    Parameters
    ----------
    bounds        : tuple or np.ndarray or None, optional, default: None
        Lower and upper bounds for the latent state, applied via
        `scaled_sigmoid`. Falls back to `DEFAULT_BOUNDS` if not provided.
    initial_prior : Prior or None, optional, default: None
        Prior used to draw the initial latent state. Falls back to
        `DEFAULT_INITIAL_PRIOR` if not provided.

    Attributes
    ----------
    bounds        : np.ndarray
        Resolved lower and upper bounds for the latent state.
    initial_prior : Prior
        Resolved prior used to draw initial latent states.
    hyper_specs   : dict
        Mapping from parameter names to either a `Prior` (sampled per batch),
        a scalar fixed value, or `None` to use a deterministic default.
        Subclasses populate this mapping.
    transition_type : str
        Broad transition category, always ``"deterministic"`` for this base.
    transition_name : str
        Short model name, such as ``"linear"``.
    """

    dtype = np.float32
    transition_type = "deterministic"

    def __init__(
        self,
        bounds: Sequence[float, float] | None = None,
        initial_prior: Prior | None = None,
    ):
        self._user_defined_bounds = bounds is not None
        self._user_defined_initial_prior = initial_prior is not None

        self.bounds = (
            np.asarray(bounds, dtype=self.dtype) if bounds is not None else np.asarray(DEFAULT_BOUNDS, dtype=self.dtype)
        )

        self.initial_prior = initial_prior if initial_prior is not None else DEFAULT_INITIAL_PRIOR
        self.hyper_specs: Dict[str, ParamSpec] = {}
        self.transition_type = "deterministic"
        self.transition_name = self.__class__.__name__

    def _resolve(self, name: str, spec: ParamSpec) -> tuple[Prior | float, bool]:
        """Resolve a single hyperparameter spec to a value and sample flag.

        Parameters
        ----------
        name : str
            Hyperparameter name, used to look up a default prior when
            `spec` is None.
        spec : Prior or float or None
            None to use the package default hyperprior, a `Prior` to
            infer the hyperparameter per-batch, or a scalar to fix it.

        Returns
        -------
        resolved : tuple - `(value, infer)`, where `value` is either
            the resolved `Prior` or the fixed float, and `infer` is
            True if `value` should be sampled rather than treated as
            fixed

        Raises
        ------
        KeyError
            If `spec` is None and no default hyperprior is registered
            for `name` in `DEFAULT_HYPER_PRIORS`.
        TypeError
            If `spec` is not None, a `Prior`, or a numeric scalar.
        """
        if spec is None:
            default = DEFAULT_DETERMINISTIC_HYPER_PRIORS.get(name)
            if default is None:
                raise KeyError(f"No default hyperprior found for '{name}'")
            if isinstance(default, Prior):
                return default, True
            return float(default), False

        if isinstance(spec, Prior):
            return spec, True

        if isinstance(spec, (float, int)):
            return float(spec), False

        raise TypeError(f"Invalid hyperparameter '{name}': {type(spec)}")

    def _sample(self, spec: Prior | float, batch_size: int) -> np.ndarray:
        """Draw batch values for a single resolved hyperparameter.

        Parameters
        ----------
        spec       : Prior or float
            A `Prior` to sample from, or a fixed scalar to broadcast.
        batch_size : int
            Number of values to produce.

        Returns
        -------
        values : np.ndarray of shape (batch_size,) - sampled or
            broadcast hyperparameter values, cast to `self.dtype`
        """
        if isinstance(spec, Prior):
            return spec.sample(batch_size).astype(self.dtype)

        return np.full(batch_size, spec, dtype=self.dtype)

    def _as_batch(self, x: np.ndarray | float, batch_size: int) -> np.ndarray:
        """Broadcast a scalar, or pass an array through, to batch shape.

        Parameters
        ----------
        x          : np.ndarray or float
            Scalar value to broadcast, or an already-batched array.
        batch_size : int
            Target batch size when `x` is a scalar.

        Returns
        -------
        values : np.ndarray of shape (batch_size,) - `x` broadcast to
            `batch_size` if scalar, otherwise `x` cast to `self.dtype`
        """
        if np.ndim(x) == 0:
            return np.full(batch_size, x, dtype=self.dtype)
        return np.asarray(x, dtype=self.dtype)

    def _bound(self, values: np.ndarray) -> np.ndarray:
        """Clip trajectory values to the configured latent-state bounds."""
        return np.clip(values, self.bounds[0], self.bounds[1]).astype(self.dtype)

    def _resolve_hyperparams(self, batch_size: int) -> tuple[Dict[str, np.ndarray], Dict[str, float]]:
        """Resolve every entry in `hyper_specs` into sampled and fixed groups.

        Parameters
        ----------
        batch_size : int
            Number of per-batch values to sample for inferred
            hyperparameters.

        Returns
        -------
        resolved : tuple - `(hyper_params, fixed_params)`, where
            `hyper_params` maps names to sampled `np.ndarray` of shape
            `(batch_size,)` and `fixed_params` maps names to fixed
            floats
        """
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
    def sample(self, batch_size: int, num_steps: int) -> Dict[str, Any]:
        """Generate `batch_size` latent trajectories of length `num_steps`.

        Parameters
        ----------
        batch_size : int
            Number of independent trajectories to draw.
        num_steps  : int
            Number of time steps per trajectory (including initial state).

        Returns
        -------
        result : dict - dictionary with keys `deterministic_params`,
            `hyper_params`, and `fixed_params`. `deterministic_params` is
            an ndarray of shape (batch_size, steps).
        """
        raise NotImplementedError
