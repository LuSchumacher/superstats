"""Backward-compatible imports for prior plotting functions."""

from .joint_prior import plot_joint_prior
from .time_invariant_prior import plot_time_invariant_prior
from .time_varying_prior import plot_time_varying_prior

__all__ = [
    "plot_joint_prior",
    "plot_time_invariant_prior",
    "plot_time_varying_prior",
]
