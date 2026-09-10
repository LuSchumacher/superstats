import numpy as np
import pytest

from superstats.prior import JointPrior, Prior
from superstats.transition import DeterministicTransition, Exponential, Linear, Logarithmic, Polynomial


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


def test_polynomial_reconstructs_trajectory_from_resolved_parameters():
    transition = Polynomial(intercept=1.0, betas=[2.0, 3.0], bounds=(-100.0, 100.0))

    trajectory = transition.sample_from_parameters(
        {"intercept": 1.0, "beta_1": 2.0, "beta_2": 3.0}, batch_size=1, num_steps=3
    )

    np.testing.assert_allclose(trajectory[0], [1.0, 2.75, 6.0])


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_polynomial_trajectory_matches_reported_hyperparameters(degree):
    transition = Polynomial(
        intercept=Prior("normal", loc=1.0, scale=0.5),
        betas=Prior("normal", loc=2.0, scale=0.5),
        degree=degree,
        bounds=(-100.0, 100.0),
    )
    result = transition.sample(batch_size=8, num_steps=10)

    reconstructed = transition.sample_from_parameters(
        {**result["hyper_params"], **result["fixed_params"]}, batch_size=8, num_steps=10
    )
    np.testing.assert_allclose(reconstructed, result["deterministic_params"])


def test_polynomial_uses_deterministic_defaults_for_all_unspecified_betas():
    result = Polynomial(degree=3).sample(batch_size=8, num_steps=10)

    assert {"beta_1", "beta_2", "beta_3"}.issubset(result["hyper_params"])
    assert not {"beta_1", "beta_2", "beta_3"}.intersection(result["fixed_params"])


def test_polynomial_reported_beta_matches_change_across_trajectory():
    transition = Polynomial(
        intercept=Prior("normal", loc=1.0, scale=0.5),
        betas=Prior("normal", loc=2.0, scale=0.5),
        degree=1,
        bounds=(-100.0, 100.0),
    )
    result = transition.sample(batch_size=8, num_steps=10)

    trajectory = result["deterministic_params"]
    np.testing.assert_allclose(result["hyper_params"]["beta_1"], trajectory[:, -1] - trajectory[:, 0], atol=1e-5)


def test_polynomial_rejects_wrong_number_of_coefficients():
    with pytest.raises(ValueError, match="degree"):
        Polynomial(betas=[1.0], degree=2)


def test_exponential_uses_intercept_and_rate():
    result = Exponential(intercept=2.0, beta=1.0, bounds=(-100.0, 100.0)).sample(batch_size=1, num_steps=3)

    np.testing.assert_allclose(result["deterministic_params"][0], 2.0 * np.exp([0.0, 0.5, 1.0]), rtol=1e-6)


def test_logarithmic_uses_intercept_and_scale():
    result = Logarithmic(intercept=1.0, beta=2.0, bounds=(-100.0, 100.0)).sample(batch_size=1, num_steps=3)

    np.testing.assert_allclose(result["deterministic_params"][0], 1.0 + 2.0 * np.log1p([0.0, 0.5, 1.0]), rtol=1e-6)


def test_new_transitions_reconstruct_sampled_trajectories():
    for transition in [Exponential(), Logarithmic()]:
        result = transition.sample(batch_size=4, num_steps=6)
        reconstructed = transition.sample_from_parameters(
            {**result["hyper_params"], **result["fixed_params"]}, batch_size=4, num_steps=6
        )
        np.testing.assert_allclose(reconstructed, result["deterministic_params"])
