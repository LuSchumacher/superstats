import numpy as np
import pytest

from superstats.prior import JointPrior, Prior
from superstats.transition.stochastic import RandomWalk

BATCH_SIZE = 6
NUM_STEPS = 10


def _build_joint_prior():
    return JointPrior(
        v=RandomWalk(),
        a=Prior("halfnormal", scale=1.0),
        tau=0.2,
        bias=0,
    )


def test_joint_prior_sample_groups_and_shapes():
    prior = _build_joint_prior()
    result = prior.sample(batch_size=BATCH_SIZE, num_steps=NUM_STEPS)

    assert set(result.keys()) == {
        "local_params",
        "deterministic_params",
        "hyper_params",
        "shared_params",
        "fixed_params",
    }

    assert result["local_params"]["v"].shape == (BATCH_SIZE, NUM_STEPS)
    assert result["deterministic_params"] == {}
    assert "v_sigma" in result["hyper_params"]
    assert result["hyper_params"]["v_sigma"].shape == (BATCH_SIZE,)

    assert result["shared_params"]["a"].shape == (BATCH_SIZE,)

    assert result["fixed_params"]["tau"] == pytest.approx(0.2)
    assert result["fixed_params"]["bias"] == 0
    assert isinstance(result["fixed_params"]["bias"], int)


def test_joint_prior_rejects_non_positive_batch_size():
    prior = _build_joint_prior()
    with pytest.raises(ValueError):
        prior.sample(batch_size=0, num_steps=NUM_STEPS)


def test_joint_prior_rejects_non_positive_num_steps():
    prior = _build_joint_prior()
    with pytest.raises(ValueError):
        prior.sample(batch_size=BATCH_SIZE, num_steps=0)


def test_joint_prior_rejects_unknown_parameter_type():
    prior = JointPrior(x=[1, 2, 3])
    with pytest.raises(TypeError):
        prior.sample(batch_size=BATCH_SIZE, num_steps=NUM_STEPS)


def test_joint_prior_param_bounds_collects_transition_bounds():
    prior = _build_joint_prior()
    bounds = prior._param_bounds()
    assert "v" in bounds
    np.testing.assert_allclose(bounds["v"], prior.params["v"].bounds)
