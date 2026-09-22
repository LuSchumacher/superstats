"""Output links from unconstrained predictors to cognitive parameters."""

from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np


class LinkFunction:
    """Transform unconstrained predictors into simulator parameters.

    Link functions can be applied by :class:`Model` at two stages. Latent
    links transform a copy of sampled parameters before parameter formulas are
    resolved. Formula links transform final formula targets after the complete
    formula has been resolved. They preserve the shape of the input.

    Parameters
    ----------
    function : {"identity", "scaled_sigmoid", "clip", "softplus", "exp"}
        or callable, optional, default: "scaled_sigmoid"
        Transformation to apply. A callable must accept a NumPy array and
        return an array with the same shape. The available named links are:

        - ``"identity"``: return the predictor unchanged.
        - ``"scaled_sigmoid"``: map the predictor smoothly into the interval
          given by ``bounds``; without explicit bounds, map it into (0, 1).
        - ``"clip"``: hard-clip the predictor to the interval given by
          ``bounds``.
        - ``"softplus"``: map the predictor smoothly into positive values.
        - ``"exp"``: exponentiate the predictor.
    bounds : sequence of two floats or None, optional, default: None
        Finite, strictly increasing lower and upper bounds. Bounds are valid
        only for ``"scaled_sigmoid"`` and ``"clip"``. When ``function`` is
        ``"scaled_sigmoid"``, ``None`` uses ``(0.0, 1.0)``. Explicit bounds
        are required for ``"clip"``.

    Attributes
    ----------
    function : str or callable
        Configured transformation.
    bounds : tuple of float or None
        Resolved interval for ``"scaled_sigmoid"`` and ``"clip"``;
        otherwise ``None``.

    Notes
    -----
    Hard clipping is applied element-wise to the completed predictor array. It
    does not feed clipped values back into a recursive transition. Latent and
    formula links operate on copies used by the simulation pipeline, while
    inference targets remain on their raw transition or coefficient scale.

    Raises
    ------
    ValueError
        If ``function`` is unknown, if bounds are invalid or supplied for an
        unsupported function, or if ``"clip"`` is used without bounds.
    """

    def __init__(
        self,
        function: Literal["identity", "scaled_sigmoid", "clip", "softplus", "exp"] | Callable = "scaled_sigmoid",
        bounds: Sequence[float] | None = None,
    ):
        if not callable(function) and function not in {
            "identity",
            "scaled_sigmoid",
            "clip",
            "softplus",
            "exp",
        }:
            raise ValueError(f"Unknown link function: {function!r}")
        self.function = function
        self.bounds = None
        if isinstance(function, str) and function in {"scaled_sigmoid", "clip"}:
            if function == "clip" and bounds is None:
                raise ValueError("bounds are required for clip.")
            interval = np.asarray((0.0, 1.0) if bounds is None else bounds, dtype=float)
            if interval.shape != (2,) or not np.all(np.isfinite(interval)) or interval[0] >= interval[1]:
                raise ValueError("bounds must contain two finite, strictly increasing values.")
            self.bounds = tuple(interval)
        elif bounds is not None:
            raise ValueError("bounds are only supported by scaled_sigmoid and clip.")

    def __call__(self, predictor):
        """Apply the configured link function.

        Parameters
        ----------
        predictor : array-like
            Scalar or array of unconstrained predictor values.

        Returns
        -------
        result : np.ndarray or NumPy scalar
            Transformed values with the same shape as ``predictor``.

        Raises
        ------
        ValueError
            If the transformation changes the predictor shape or produces a
            non-finite value.
        """
        x = np.asarray(predictor)
        if callable(self.function):
            result = np.asarray(self.function(x))
        elif self.function == "identity":
            result = x
        elif self.function == "softplus":
            result = np.logaddexp(0.0, x)
        elif self.function == "exp":
            result = np.exp(x)
        elif self.function == "clip":
            lower, upper = self.bounds
            result = np.clip(x, lower, upper)
        else:
            # Stable for large positive and negative predictors.
            result = np.exp(-np.logaddexp(0.0, -x))
            if self.function == "scaled_sigmoid":
                lower, upper = self.bounds
                result = lower + (upper - lower) * result
        if result.shape != x.shape:
            raise ValueError(f"Link function changed shape from {x.shape} to {result.shape}.")
        if not np.all(np.isfinite(result)):
            raise ValueError("Link function produced non-finite parameter values.")
        return result
