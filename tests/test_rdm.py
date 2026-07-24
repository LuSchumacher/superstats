from collections.abc import Mapping

import numpy as np

from superstats.simulation import sample_rdm


def test_sample_rdm_shape_and_dtype():
    num_trials = 25
    v_base = np.random.normal(0.0, 1.0, size=num_trials).astype(np.float32)
    v_diff = np.random.normal(0.0, 1.0, size=num_trials).astype(np.float32)
    a_base = np.full(num_trials, 1.0, dtype=np.float32)
    tau = np.full(num_trials, 0.2, dtype=np.float32)
    bias = np.full(num_trials, 1.0, dtype=np.float32)
    sigma_diff = np.full(num_trials, 1.0, dtype=np.float32)

    data = sample_rdm(v_base, v_diff, a_base, tau, bias, sigma_diff)

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


def test_sample_rdm_correct_accumulator_wins_with_strong_drift_advantage():
    num_trials = 5
    v_base = np.zeros(num_trials, dtype=np.float32)
    v_diff = np.full(num_trials, 1e6, dtype=np.float32)  # huge advantage to correct
    a_base = np.full(num_trials, 0.001, dtype=np.float32)
    tau = np.zeros(num_trials, dtype=np.float32)
    bias = np.full(num_trials, 1.0, dtype=np.float32)
    sigma_diff = np.full(num_trials, 1.0, dtype=np.float32)
    correct_idx = np.ones(num_trials, dtype=np.float32)  # accumulator 1 is correct

    data = sample_rdm(v_base, v_diff, a_base, tau, bias, sigma_diff, correct_idx=correct_idx)

    # the correct accumulator has an overwhelming drift advantage, so it should win
    assert np.all(data["choice"] == 1.0)
    assert np.all(data["response_time"] > 0)


def test_sample_rdm_times_out_when_max_steps_too_small():
    num_trials = 5
    v_base = np.zeros(num_trials, dtype=np.float32)
    v_diff = np.zeros(num_trials, dtype=np.float32)
    a_base = np.full(num_trials, 1e6, dtype=np.float32)
    tau = np.zeros(num_trials, dtype=np.float32)
    bias = np.full(num_trials, 1.0, dtype=np.float32)
    sigma_diff = np.full(num_trials, 1.0, dtype=np.float32)

    data = sample_rdm(v_base, v_diff, a_base, tau, bias, sigma_diff, max_steps=1)

    assert np.all(data["response_time"] == -1.0)
    assert np.all(data["choice"] == -1.0)


def test_sample_rdm_preserves_trialwise_parameter_alignment():
    # A zero-noise, zero-drift process with a tiny boundary should always
    # time out, independently for every trial.
    n = 4
    data = sample_rdm(
        np.zeros(n, dtype=np.float32),
        np.zeros(n, dtype=np.float32),
        np.full(n, 1e6, dtype=np.float32),
        np.zeros(n, dtype=np.float32),
        np.full(n, 0.5, dtype=np.float32),
        np.zeros(n, dtype=np.float32),
        max_steps=2,
    )

    assert np.array_equal(data["response_time"], np.full(n, -1.0, dtype=np.float32))
    assert np.array_equal(data["choice"], np.full(n, -1.0, dtype=np.float32))
