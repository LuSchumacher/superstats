import superstats as sup

from superstats.prior import JointPrior, Prior
from superstats.simulation import Model
from superstats.workflow import Workflow


def test_top_level_shortcuts_expose_core_interfaces():
    assert sup.JointPrior is JointPrior
    assert sup.Prior is Prior
    assert sup.Model is Model
    assert sup.Workflow is Workflow
