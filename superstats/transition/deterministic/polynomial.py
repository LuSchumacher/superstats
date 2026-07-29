"""Polynomial deterministic transition."""

from collections.abc import Sequence
from typing import Any, Dict

import numpy as np

from .deterministic_transition import DeterministicTransition, Prior


class Polynomial(DeterministicTransition):
    """Deterministic polynomial transition with an intercept and beta weights.

    Parameters
    ----------
    bounds          : sequence of two floats or None, optional, default: None
        Lower and upper bounds for the deterministic trajectory. Tuples and
        lists are accepted.
    intercept       : float, Prior, or None, optional, default: None
        Constant term of the polynomial. A `Prior` samples one intercept per
        trajectory; `None` uses the deterministic default prior.
    betas           : float, Prior, sequence of float/Prior/None, or None
        Polynomial coefficients for the non-constant terms. If a single scalar
        or `Prior` is provided, the same specification is used for every beta.
        If a sequence is provided, it must have length `degree` and
        each element is used for the corresponding beta weight.
    degree : int, optional, default: 2
        Number of polynomial terms beyond the intercept. For example,
        `degree=1` reproduces a linear model, while the default
        `degree=2` gives a quadratic model.
    normalize_steps : bool, optional, default: True
        If `True`, use a time axis from 0 to 1. If `False`, use integer step
        indices, so higher-order terms are evaluated on raw step numbers.

    Notes
    -----
    The `sample` method returns a dict with keys `deterministic_params`,
    `hyper_params`, and `fixed_params`. Trajectory values are clipped to
    `bounds`.
    """

    def __init__(
        self,
        bounds: Sequence[float, float] | None = None,
        intercept: float | Prior | None = None,
        betas: float | Prior | Sequence[float | Prior | None] | None = None,
        degree: int = 2,
        normalize_steps: bool = True,
    ):
        super().__init__(bounds=bounds)

        if degree < 1:
            raise ValueError("degree must be at least 1")

        self.degree = degree
        self.normalize_steps = normalize_steps

        self.hyper_specs = {"intercept": intercept}

        beta_specs = self._expand_beta_specs(betas)
        for idx, spec in enumerate(beta_specs, start=1):
            self.hyper_specs[f"beta_{idx}"] = spec

        self.transition_name = "polynomial"

    def _expand_beta_specs(
        self,
        betas: float | Prior | Sequence[float | Prior | None] | None,
    ) -> Sequence[float | Prior | None]:
        """Normalize beta input into one spec per polynomial term."""
        if isinstance(betas, (Prior, float, int)) or betas is None:
            return [betas] * self.degree

        if isinstance(betas, Sequence) and not isinstance(betas, (str, bytes)):
            if len(betas) != self.degree:
                raise ValueError(f"betas must have length equal to degree ({self.degree}), got {len(betas)}")
            return betas

        raise TypeError("betas must be a scalar, Prior, None, or a sequence of scalars/Priors")

    def _resolve_beta(self, spec: float | Prior | None) -> tuple[Prior | float, bool]:
        """Resolve a beta spec, defaulting to a fixed zero if unspecified."""
        if spec is None:
            return 0.0, False
        if isinstance(spec, Prior):
            return spec, True
        if isinstance(spec, (float, int)):
            return spec, False
        raise TypeError(f"Invalid beta specification: {type(spec)}")

    def _resolve_hyperparams(self, batch_size: int) -> tuple[Dict[str, np.ndarray], Dict[str, float]]:
        """Resolve intercept and beta specs into sampled and fixed groups."""
        hyper_params = {}
        fixed_params = {}

        for name, spec in self.hyper_specs.items():
            if name == "intercept":
                value, infer = self._resolve(name, spec)
            else:
                value, infer = self._resolve_beta(spec)

            if infer:
                hyper_params[name] = self._sample(value, batch_size)
            else:
                fixed_params[name] = value

        return hyper_params, fixed_params

    def sample(self, batch_size: int, num_steps: int) -> Dict[str, Any]:
        """Draw `batch_size` polynomial trajectories of length `num_steps`."""
        hyper, fixed = self._resolve_hyperparams(batch_size)

        resolved = {
            name: hyper[name] if name in hyper else np.full(batch_size, fixed[name], dtype=self.dtype)
            for name in self.hyper_specs
        }

        if self.normalize_steps:
            index = np.linspace(0.0, 1.0, num_steps, dtype=self.dtype)
        else:
            index = np.arange(num_steps, dtype=self.dtype)

        local = np.repeat(resolved["intercept"][:, None], num_steps, axis=1)

        for power in range(1, self.degree + 1):
            local += resolved[f"beta_{power}"][:, None] * np.power(index[None, :], power)

        return {
            "deterministic_params": self._bound(local),
            "hyper_params": hyper,
            "fixed_params": fixed,
        }

    def sample_from_parameters(
        self,
        params: Dict[str, np.ndarray | float],
        batch_size: int,
        num_steps: int,
    ) -> np.ndarray:
        """Generate trajectories from resolved intercept and beta terms."""
        intercept = np.broadcast_to(np.asarray(params["intercept"], dtype=self.dtype), (batch_size,))
        if self.normalize_steps:
            index = np.linspace(0.0, 1.0, num_steps, dtype=self.dtype)
        else:
            index = np.arange(num_steps, dtype=self.dtype)

        trajectory = np.repeat(intercept[:, None], num_steps, axis=1)
        for power in range(1, self.degree + 1):
            beta = np.broadcast_to(np.asarray(params[f"beta_{power}"], dtype=self.dtype), (batch_size,))
            trajectory += beta[:, None] * np.power(index[None, :], power)

        return self._bound(trajectory)
