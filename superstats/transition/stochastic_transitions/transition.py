"""Compatibility shim for the old transition import path."""

from .stochastic_transition import ParamSpec, Prior, StochasticTransition

__all__ = ["ParamSpec", "Prior", "StochasticTransition"]
