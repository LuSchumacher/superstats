from collections.abc import Mapping

import numpy as np
import pytest

from superstats.simulation import sample_rdm


def test_sample_rdm_shape_and_dtype():
    num_trials = 25
    v_base = np.random.normal(0.0, 1.0, size=num_trials).astype(np.float32)
    v_diff = np.random.normal(0.0, 1.0, size=num_trials).astype(np.float32)
    a_base = np.full(num_trials, 1.0, dtype=np.float32)
    a_diff = np.zeros(num_trials, dtype=np.float32)
    tau = np.full(num_trials, 0.2, dtype=np.float32)
    sigma_diff = np.zeros(num_trials, dtype=np.float32)

    data = sample_rdm(v_base, v_diff, a_base, a_diff, tau, sigma_diff)

    assert isinstance(data, Mapping)
    assert set(data) == {"response_time", "choice"}
    assert data["response_time"].shape == (num_trials,)
    assert data["choice"].shape == (num_trials,)
    assert data["response_time"].dtype == np.float32
    assert data["choice"].dtype == np.float32

    rt, choice = data["response_time"], data["choice"]
    # response times are either positive (a decision was reached) or -1.0 (timeout)
    assert np.all((rt > 0) | (rt == -1.0))
    # winner is a valid accumulator index (0 or 1 with the default of 2) or -1.0
    assert np.all(np.isin(choice, [-1.0, 0.0, 1.0]))


def test_sample_rdm_boosted_accumulator_wins_with_strong_drift_advantage():
    num_trials = 5
    v_base = np.zeros(num_trials, dtype=np.float32)
    v_diff = np.full(num_trials, 1e6, dtype=np.float32)
    a_base = np.full(num_trials, 0.001, dtype=np.float32)
    a_diff = np.zeros(num_trials, dtype=np.float32)
    tau = np.zeros(num_trials, dtype=np.float32)
    sigma_diff = np.zeros(num_trials, dtype=np.float32)
    boost_idx = np.array([1, 0, 1, 0, 1], dtype=np.int32)

    data = sample_rdm(v_base, v_diff, a_base, a_diff, tau, sigma_diff, boost_idx=boost_idx, sigma_base=0.0)

    np.testing.assert_array_equal(data["choice"], boost_idx)
    expected_rt = np.logaddexp(0.0, a_base) / np.logaddexp(0.0, v_base + v_diff / 2)
    np.testing.assert_allclose(data["response_time"], expected_rt, rtol=1e-6)


def test_sample_rdm_times_out_when_max_steps_too_small():
    num_trials = 5
    v_base = np.zeros(num_trials, dtype=np.float32)
    v_diff = np.zeros(num_trials, dtype=np.float32)
    a_base = np.full(num_trials, 1e6, dtype=np.float32)
    a_diff = np.zeros(num_trials, dtype=np.float32)
    tau = np.zeros(num_trials, dtype=np.float32)
    sigma_diff = np.zeros(num_trials, dtype=np.float32)

    data = sample_rdm(v_base, v_diff, a_base, a_diff, tau, sigma_diff, sigma_base=0.0, max_steps=1)

    assert np.all(data["response_time"] == -1.0)
    assert np.all(data["choice"] == -1.0)


def test_sample_rdm_preserves_trialwise_parameter_alignment():
    # Equal drift and boundary predictors give a decision time of exactly one
    # second; the trial-specific nondecision times must remain aligned.
    n = 4
    predictors = np.array([-2.0, -0.5, 0.5, 2.0], dtype=np.float32)
    tau = np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float32)
    data = sample_rdm(
        predictors,
        np.zeros(n, dtype=np.float32),
        predictors,
        np.zeros(n, dtype=np.float32),
        tau,
        np.zeros(n, dtype=np.float32),
        sigma_base=0.0,
        dt=2.0,
        max_steps=1,
    )

    np.testing.assert_allclose(data["response_time"], tau + 1.0)
    np.testing.assert_array_equal(data["choice"], np.zeros(n, dtype=np.float32))


@pytest.mark.parametrize("contrast", [-2.0, 2.0])
def test_sample_rdm_boundary_bias_is_independent_of_drift_boost(contrast):
    bias_idx = np.array([0, 1], dtype=np.int32)
    data = sample_rdm(
        v_base=np.full(2, 2.0, dtype=np.float32),
        v_diff=np.zeros(2, dtype=np.float32),
        a_base=np.zeros(2, dtype=np.float32),
        a_diff=np.full(2, contrast, dtype=np.float32),
        tau=np.full(2, 0.2, dtype=np.float32),
        sigma_diff=np.zeros(2, dtype=np.float32),
        bias_idx=bias_idx,
        sigma_base=0.0,
        dt=1.0,
        max_steps=1,
    )

    expected_choice = bias_idx if contrast < 0 else 1 - bias_idx
    np.testing.assert_array_equal(data["choice"], expected_choice)
    expected_rt = 0.2 + np.logaddexp(0.0, -1.0) / np.logaddexp(0.0, 2.0)
    np.testing.assert_allclose(data["response_time"], expected_rt)
