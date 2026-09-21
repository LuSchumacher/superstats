"""Tools for neural superstatistics and dynamic Bayesian estimation."""

import logging as _logging

from . import prior, diagnostics, simulation, workflow, approximators
from .approximators import CompositeApproximator, MarginalApproximator, JointApproximator
from .prior import JointPrior, Prior
from .simulation import Model, LinkFunction
from .workflow import Workflow
from .simulation import ContextSimulator, Formula

_logging.basicConfig(level=_logging.INFO)
_logging.getLogger(__name__).setLevel(_logging.INFO)


__all__ = [
    "LinkFunction",
    "diagnostics",
    "prior",
    "simulation",
    "workflow",
    "approximators",
    "CompositeApproximator",
    "MarginalApproximator",
    "JointApproximator",
    "JointPrior",
    "Prior",
    "Model",
    "Workflow",
    "ContextSimulator",
    "Formula",
]

del _logging
