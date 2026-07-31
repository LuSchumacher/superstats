"""Backward-compatible imports for posterior plotting functions."""

from .time_invariant_posterior import plot_time_invariant_posterior
from .time_varying_posterior import plot_time_varying_posterior

__all__ = [
    "plot_time_invariant_posterior",
    "plot_time_varying_posterior",
]
