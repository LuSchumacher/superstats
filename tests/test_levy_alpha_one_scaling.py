import numpy as np
from numba import njit
from scipy import stats

from superstats.prior import Prior
from superstats.transition.stochastic import LevyFlight
from superstats.transition.stochastic.levy_flight import _sample_alpha_stable


@njit
def _sample_seeded(seed: int, n: int, alpha: float, beta: float, scale: float) -> np.ndarray:
    np.random.seed(seed)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = _sample_alpha_stable(alpha, beta, scale)
    return out


def test_levy_alpha_one_scaling_matches_s1_cdf():
    alpha, beta, scale = 1.0, 0.8, 3.0
    probs = np.array([0.25, 0.5, 0.75])
    draws = _sample_seeded(0, 100_000, alpha, beta, scale)
    observed_quantiles = np.quantile(draws, probs)

    previous = stats.levy_stable.parameterization
    try:
        stats.levy_stable.parameterization = "S1"
        cdf_at_quantiles = stats.levy_stable.cdf(observed_quantiles, alpha, beta, loc=0.0, scale=scale)
    finally:
        stats.levy_stable.parameterization = previous

    np.testing.assert_allclose(cdf_at_quantiles, probs, atol=0.01)


def test_levy_alpha_one_satisfies_s1_stability_additivity():
    alpha, beta = 1.0, 0.8
    y1 = _sample_seeded(1, 100_000, alpha, beta, 1.0)
    y2 = _sample_seeded(2, 100_000, alpha, beta, 1.0)
    z2 = _sample_seeded(3, 100_000, alpha, beta, 2.0)

    np.testing.assert_allclose(np.median(y1 + y2), np.median(z2), atol=0.05)


def test_levy_alpha_one_zero_scale_is_deterministic_and_finite():
    transition = LevyFlight(
        initial_prior=Prior("normal", loc=1.0, scale=0.0),
        sigma=0.0,
        delta=2.0,
        alpha=1.0,
        beta=0.8,
    )

    trajectory = transition.sample(batch_size=2, num_steps=4)["local_params"]
    expected = np.broadcast_to(1.0 + 2.0 * np.arange(4, dtype=np.float32), (2, 4))
    np.testing.assert_allclose(trajectory, expected, rtol=1e-6)
