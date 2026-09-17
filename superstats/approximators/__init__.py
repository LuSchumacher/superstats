"""Composite posterior approximators for dynamic and invariant parameters."""

from .composite_approximator import CompositeApproximator
from .marginal_approximator import MarginalApproximator
from .joint_approximator import JointApproximator

__all__ = ["CompositeApproximator", "MarginalApproximator", "JointApproximator"]
