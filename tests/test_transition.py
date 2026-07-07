import numpy as np
import pytest

from superstats.prior import Prior
from superstats.transition import (
    AutoRegression,
    Jump,
    Mixture,
    OrnsteinUhlenbeck,
    RandomWalk,
)

BATCH_SIZE = 8
NUM_STEPS = 12


@pytest.mark.parametrize(
    "transition, expected_hyper_keys, expected_fixed_keys",
    [
        (RandomWalk(), {"sigma"}, {"delta"}),
        (AutoRegression(), {"sigma", "phi"}, {"delta"}),
        (OrnsteinUhlenbeck(), {"sigma", "mu", "theta"}, set()),
        (Jump(), set(), {"p_jump"}),
    ],
)
def test_transition_sample_shape_and_keys(transition, expected_hyper_keys, expected_fixed_keys):
    result = transition.sample(batch_size=BATCH_SIZE, num_steps=NUM_STEPS)

    assert set(result.keys()) == {"local_params", "hyper_params", "fixed_params"}

    local_params = result["local_params"]
    assert isinstance(local_params, np.ndarray)
    assert local_params.shape == (BATCH_SIZE, NUM_STEPS)
    assert local_params.dtype == np.float32

    # values must respect the (default) bounds
    lower, upper = transition.bounds
    assert np.all(local_params >= lower - 1e-4)
    assert np.all(local_params <= upper + 1e-4)

    assert set(result["hyper_params"].keys()) == expected_hyper_keys
    for values in result["hyper_params"].values():
        assert values.shape == (BATCH_SIZE,)

    assert set(result["fixed_params"].keys()) == expected_fixed_keys


def test_random_walk_sample_one_step_returns_finite_float():
    transition = RandomWalk()
    x_next = transition.sample_one_step(x=0.0, params={"sigma": 0.1, "delta": 0.0})
    assert isinstance(x_next, float)
    assert np.isfinite(x_next)


def test_auto_regression_sample_one_step_returns_finite_float():
    transition = AutoRegression()
    x_next = transition.sample_one_step(x=0.0, params={"sigma": 0.1, "phi": 0.9, "delta": 0.0})
    assert isinstance(x_next, float)
    assert np.isfinite(x_next)


def test_ornstein_uhlenbeck_sample_one_step_returns_finite_float():
    transition = OrnsteinUhlenbeck()
    x_next = transition.sample_one_step(x=0.0, params={"mu": 0.0, "theta": 0.1, "sigma": 0.1})
    assert isinstance(x_next, float)
    assert np.isfinite(x_next)


def test_jump_sample_one_step_returns_finite_float():
    transition = Jump()
    x_next = transition.sample_one_step(x=0.0, params={"p_jump": 1.0})
    assert isinstance(x_next, float)
    assert np.isfinite(x_next)


def test_mixture_sample_shape_and_keys():
    mixture = Mixture(
        transitions=[RandomWalk(), Jump()],
        bounds=(-3.0, 3.0),
        initial_prior=Prior("normal", loc=0.0, scale=1.0),
    )

    result = mixture.sample(batch_size=BATCH_SIZE, num_steps=NUM_STEPS)

    assert set(result.keys()) == {"local_params", "regimes", "hyper_params", "fixed_params"}

    local_params = result["local_params"]
    assert local_params.shape == (BATCH_SIZE, NUM_STEPS)
    assert np.all(local_params >= -3.0 - 1e-4)
    assert np.all(local_params <= 3.0 + 1e-4)

    regimes = result["regimes"]
    assert regimes.shape == (BATCH_SIZE, NUM_STEPS)
    assert set(np.unique(regimes)).issubset({0, 1})

    assert "rw_sigma" in result["hyper_params"]
    assert "rw_delta" in result["fixed_params"]
    assert "jump_p_jump" in result["fixed_params"]
    assert "mixture_weights" in result["fixed_params"]


def test_mixture_requires_at_least_two_transitions():
    with pytest.raises(ValueError):
        Mixture(transitions=[RandomWalk()])


def test_mixture_rejects_component_with_own_bounds():
    with pytest.raises(ValueError):
        Mixture(transitions=[RandomWalk(bounds=(-1.0, 1.0)), Jump()])


def test_mixture_with_dirichlet_weights():
    mixture = Mixture(
        transitions=[RandomWalk(), Jump()],
        mixture_weights=Prior("dirichlet", alpha=[5.0, 5.0]),
        bounds=(-3.0, 3.0),
        initial_prior=Prior("normal", loc=0.0, scale=1.0),
    )
    result = mixture.sample(batch_size=BATCH_SIZE, num_steps=NUM_STEPS)
    assert result["hyper_params"]["mixture_weights"].shape == (BATCH_SIZE, 2)
