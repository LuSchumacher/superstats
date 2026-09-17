"""Output links from unconstrained predictors to cognitive parameters."""

from collections.abc import Callable, Sequence

import numpy as np


class LinkFunction:
    """A shape-preserving output link, applied by :class:`Model`.

    ``LinkFunction(bounds=(lower, upper))`` uses a scaled sigmoid. Other
    supported functions are ``identity``, ``sigmoid``, ``softplus``, ``exp``,
    or a callable accepting a NumPy array. Bounds are only valid for the
    scaled sigmoid. The default interval is (0, 1).
    """

    def __init__(
        self,
        function: str | Callable = "scaled_sigmoid",
        bounds: Sequence[float] | None = None,
    ):
        if not callable(function) and function not in {"identity", "sigmoid", "scaled_sigmoid", "softplus", "exp"}:
            raise ValueError(f"Unknown link function: {function!r}")
        self.function = function
        self.bounds = None
        if isinstance(function, str) and function == "scaled_sigmoid":
            interval = np.asarray((0.0, 1.0) if bounds is None else bounds, dtype=float)
            if interval.shape != (2,) or not np.all(np.isfinite(interval)) or interval[0] >= interval[1]:
                raise ValueError("bounds must contain two finite, strictly increasing values.")
            self.bounds = tuple(interval)
        elif bounds is not None:
            raise ValueError("bounds are only supported by scaled_sigmoid.")
        elif isinstance(function, str) and function == "sigmoid":
            self.bounds = (0.0, 1.0)

    def __call__(self, predictor):
        x = np.asarray(predictor)
        if callable(self.function):
            result = np.asarray(self.function(x))
        elif self.function == "identity":
            result = x
        elif self.function == "softplus":
            result = np.logaddexp(0.0, x)
        elif self.function == "exp":
            result = np.exp(x)
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
