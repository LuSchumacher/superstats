"""Deterministic transition models for latent time-varying parameters."""

from .deterministic_transition import DeterministicTransition
from .linear import Linear

__all__ = ["DeterministicTransition", "Linear"]
