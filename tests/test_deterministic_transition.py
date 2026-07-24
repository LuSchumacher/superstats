import numpy as np

from superstats.prior import JointPrior
from superstats.transition import DeterministicTransition, Linear


def test_linear_returns_standard_parameter_groups():
    transition = Linear(intercept=1.0, beta=0.5)
    result = transition.sample(batch_size=4, num_steps=6)

    assert isinstance(transition, DeterministicTransition)
    assert result["deterministic_params"].shape == (4, 6)
    assert result["hyper_params"] == {}
    assert set(result["fixed_params"]) == {"intercept", "beta"}
    assert np.all(result["deterministic_params"] == result["deterministic_params"][0])


def test_linear_uses_deterministic_defaults_as_inferred_priors():
    transition = Linear()
    result = transition.sample(batch_size=4, num_steps=6)

    assert set(result["hyper_params"]) == {"intercept", "beta"}
    assert result["hyper_params"]["intercept"].shape == (4,)
    assert result["hyper_params"]["beta"].shape == (4,)
    assert np.all(np.isfinite(result["deterministic_params"]))


def test_linear_trajectory_is_deterministic_given_fixed_parameters():
    transition = Linear(intercept=0.0, beta=2.0, bounds=(0.0, 2.0))
    result = transition.sample(batch_size=1, num_steps=5)

    expected = np.linspace(0.0, 2.0, 5)
    np.testing.assert_allclose(result["deterministic_params"][0], expected)


def test_linear_can_use_step_based_slope():
    result = Linear(intercept=0.0, beta=1.0, bounds=(0.0, 4.0), normalize_steps=False).sample(batch_size=1, num_steps=4)

    np.testing.assert_allclose(result["deterministic_params"][0], np.arange(4.0))


def test_linear_clips_initial_value_to_bounds():
    result = Linear(intercept=-5.0, beta=0.0, bounds=(0.2, 4.0)).sample(batch_size=1, num_steps=3)

    np.testing.assert_allclose(result["deterministic_params"], 0.2)


def test_joint_prior_accepts_deterministic_transitions():
    result = JointPrior(x=Linear(intercept=0.0, beta=1.0)).sample(batch_size=3, num_steps=4)

    assert result["deterministic_params"]["x"].shape == (3, 4)
    assert "x_intercept" in result["fixed_params"]
