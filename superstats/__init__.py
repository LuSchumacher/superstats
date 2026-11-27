"""Tools for neural superstatistics and dynamic Bayesian estimation."""

import logging as _logging

from . import prior, diagnostics, simulation, workflow, estimators
from .estimators import CompositeEstimator, MarginalEstimator, JointEstimator
from .prior import JointPrior, Prior
from .simulation import Model, LinkFunction
from .workflow import Workflow
from .simulation import Categorical, ContextSimulator, Formula

_logging.basicConfig(level=_logging.INFO)
_logging.getLogger(__name__).setLevel(_logging.INFO)


__all__ = [
    "LinkFunction",
    "diagnostics",
    "prior",
    "simulation",
    "workflow",
    "estimators",
    "CompositeEstimator",
    "MarginalEstimator",
    "JointEstimator",
    "JointPrior",
    "Prior",
    "Model",
    "Workflow",
    "ContextSimulator",
    "Formula",
    "Categorical",
]

del _logging
