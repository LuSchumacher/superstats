import numpy as np
import pytest

from superstats.simulation import sample_cpt


def _parameters():
    return (np.full(8, 0.8), np.full(8, 2.0), np.full(8, 1.0), np.full(8, 0.7))


@pytest.mark.parametrize("num_outcomes", [1, 2, 3, 4, 5, 10])
def test_sample_cpt_supports_arbitrary_outcome_count(num_outcomes):
    outcomes_a = np.zeros((8, num_outcomes))
    outcomes_b = np.ones((8, num_outcomes))
    probabilities_a = np.full_like(outcomes_a, 1.0 / num_outcomes)
    probabilities_b = np.full_like(outcomes_b, 1.0 / num_outcomes)
    alpha, lambda_, tau, gamma = _parameters()

    result = sample_cpt(
        alpha,
        lambda_,
        tau,
        gamma,
        outcomes_a,
        outcomes_b,
        probabilities_a,
        probabilities_b,
    )

    assert result["choice"].shape == (8,)
    assert np.all(np.isin(result["choice"], [0, 1]))


def test_sample_cpt_supports_different_outcome_counts_and_sure_outcome():
    outcomes_a = np.full((8, 1), 2.0)
    probabilities_a = np.ones((8, 1))
    outcomes_b = np.tile(np.array([[0.0, 1.0, 3.0]]), (8, 1))
    probabilities_b = np.full((8, 3), 1.0 / 3.0)

    result = sample_cpt(
        np.ones(8),
        np.ones(8),
        np.ones(8),
        np.ones(8),
        outcomes_a,
        outcomes_b,
        probabilities_a,
        probabilities_b,
    )

    assert result["choice"].shape == (8,)
    assert np.all(np.isin(result["choice"], [0, 1]))


def test_sample_cpt_rejects_mismatched_probability_shapes():
    with pytest.raises(ValueError, match="same shape"):
        sample_cpt(
            np.ones(2),
            np.ones(2),
            np.ones(2),
            np.ones(2),
            np.ones((2, 1)),
            np.ones((2, 1)),
            np.ones((2, 2)),
            np.ones((2, 1)),
        )
