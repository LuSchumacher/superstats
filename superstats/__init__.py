"""Tools for neural superstatistics and dynamic Bayesian estimation."""

import logging as _logging

from . import prior, diagnostics, simulation, workflow, networks
from .prior import JointPrior, Prior
from .simulation import Model
from .workflow import Workflow

_logging.basicConfig(level=_logging.INFO)
_logging.getLogger(__name__).setLevel(_logging.INFO)

__all__ = [
    "diagnostics",
    "prior",
    "simulation",
    "workflow",
    "networks",
    "JointPrior",
    "Prior",
    "Model",
    "Workflow",
]

del _logging
