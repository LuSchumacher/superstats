import numpy as np
import pytest

from superstats.prior import Prior
from superstats.transition.stochastic_transitions import LevyFlight, OrnsteinUhlenbeck


@pytest.mark.parametrize(
    "transition",
    [OrnsteinUhlenbeck(sigma=0.0, mu=0.0, theta=1.0), LevyFlight(sigma=0.0, alpha=2.0)],
)
def test_additional_transitions_have_finite_expected_shapes(transition):
    result = transition.sample(batch_size=3, num_steps=1)

    assert result["local_params"].shape == (3, 1)
    assert np.all(np.isfinite(result["local_params"]))


def test_ou_zero_noise_reverts_towards_mean():
    transition = OrnsteinUhlenbeck(initial_prior=Prior("normal", loc=2.0, scale=0.0), sigma=0.0, mu=0.0, theta=0.5)

    trajectory = transition.sample(1, 4)["local_params"][0]

    latent = np.array([2.0, 1.0, 0.5, 0.25])
    np.testing.assert_allclose(trajectory, 1.0 / (1.0 + np.exp(-latent)), rtol=1e-6)


def test_levy_zero_noise_and_drift_is_deterministic():
    transition = LevyFlight(initial_prior=Prior("normal", loc=1.0, scale=0.0), sigma=0.0, delta=2.0, alpha=1.5)

    latent = 1.0 + 2.0 * np.arange(4)
    np.testing.assert_allclose(transition.sample(1, 4)["local_params"][0], 1.0 / (1.0 + np.exp(-latent)), rtol=1e-6)


def test_levy_sample_one_step_defaults_beta_to_symmetric_noise():
    np.random.seed(1)
    value = LevyFlight().sample_one_step(2.0, {"sigma": 0.0, "delta": 3.0, "alpha": 1.5})

    assert value == 5.0
