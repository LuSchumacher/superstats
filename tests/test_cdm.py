from collections.abc import Mapping

import numpy as np

from superstats.simulation import sample_cdm


def test_sample_cdm_shape_and_dtype():
    num_trials = 25
    v_angle = np.random.uniform(-np.pi, np.pi, size=num_trials).astype(np.float32)
    v_length = np.random.uniform(0.0, 2.0, size=num_trials).astype(np.float32)
    a = np.full(num_trials, 1.0, dtype=np.float32)
    tau = np.full(num_trials, 0.2, dtype=np.float32)

    data = sample_cdm(v_angle, v_length, a, tau)

    assert isinstance(data, Mapping)
    assert set(data) == {"response_time", "choice"}
    assert data["response_time"].shape == (num_trials,)
    assert data["choice"].shape == (num_trials,)
    assert data["response_time"].dtype == np.float64
    assert data["choice"].dtype == np.float64

    rt, angle = data["response_time"], data["choice"]
    # response times are either positive (a decision was reached) or -5.0 (timeout)
    assert np.all((rt > 0) | (rt == -5.0))
    # response angle is within [-pi, pi] (arctan2 range) or -5.0 on timeout
    assert np.all(((angle >= -np.pi) & (angle <= np.pi)) | (angle == -5.0))


def test_sample_cdm_crosses_near_drift_direction_with_strong_drift():
    num_trials = 5
    target_angle = 1.0
    v_angle = np.full(num_trials, target_angle, dtype=np.float32)
    v_length = np.full(num_trials, 1e6, dtype=np.float32)  # overwhelming drift
    a = np.full(num_trials, 0.001, dtype=np.float32)
    tau = np.zeros(num_trials, dtype=np.float32)

    data = sample_cdm(v_angle, v_length, a, tau)

    # with drift dominating the noise, the crossing angle should match v_angle
    assert np.all(data["response_time"] > 0)
    assert np.allclose(data["choice"], target_angle, atol=1e-2)


def test_sample_cdm_times_out_when_max_steps_too_small():
    num_trials = 5
    v_angle = np.zeros(num_trials, dtype=np.float32)
    v_length = np.zeros(num_trials, dtype=np.float32)
    a = np.full(num_trials, 1e6, dtype=np.float32)
    tau = np.zeros(num_trials, dtype=np.float32)

    data = sample_cdm(v_angle, v_length, a, tau, max_steps=1)

    assert np.all(data["response_time"] == -5.0)
    assert np.all(data["choice"] == -5.0)
