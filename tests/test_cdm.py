import numpy as np

from superstats.simulation import sample_cdm


def test_sample_cdm_shape_and_dtype():
    num_trials = 25
    v_angle = np.random.uniform(-np.pi, np.pi, size=num_trials).astype(np.float32)
    v_length = np.random.uniform(0.0, 2.0, size=num_trials).astype(np.float32)
    a = np.full(num_trials, 1.0, dtype=np.float32)
    tau = np.full(num_trials, 0.2, dtype=np.float32)

    data = sample_cdm(v_angle, v_length, a, tau)

    assert isinstance(data, np.ndarray)
    assert data.shape == (num_trials, 2)
    assert data.dtype == np.float64

    rt, angle = data[:, 0], data[:, 1]
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
    assert np.all(data[:, 0] > 0)
    assert np.allclose(data[:, 1], target_angle, atol=1e-2)


def test_sample_cdm_times_out_when_max_steps_too_small():
    num_trials = 5
    v_angle = np.zeros(num_trials, dtype=np.float32)
    v_length = np.zeros(num_trials, dtype=np.float32)
    a = np.full(num_trials, 1e6, dtype=np.float32)
    tau = np.zeros(num_trials, dtype=np.float32)

    data = sample_cdm(v_angle, v_length, a, tau, max_steps=1)

    assert np.all(data[:, 0] == -5.0)
    assert np.all(data[:, 1] == -5.0)
