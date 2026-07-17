from collections.abc import Mapping

import numpy as np

from superstats.simulation import sample_ddm


def test_sample_ddm_shape_and_dtype():
    num_trials = 25
    v = np.random.normal(0.0, 1.0, size=num_trials).astype(np.float32)
    a = np.full(num_trials, 1.0, dtype=np.float32)
    tau = np.full(num_trials, 0.2, dtype=np.float32)
    bias = np.zeros(num_trials, dtype=np.float32)

    data = sample_ddm(v, a, tau, bias)

    assert isinstance(data, Mapping)
    assert set(data) == {"response_time", "choice"}
    assert data["response_time"].shape == (num_trials,)
    assert data["choice"].shape == (num_trials,)
    assert data["response_time"].dtype == np.float32
    assert data["choice"].dtype == np.float32

    rt, choice = data["response_time"], data["choice"]
    # response times are either positive (a decision was reached) or -1.0 (timeout)
    assert np.all((rt > 0) | (rt == -1.0))
    assert np.all(np.isin(choice, [-1.0, 0.0, 1.0]))


def test_sample_ddm_hits_upper_boundary_with_strong_positive_drift():
    num_trials = 5
    v = np.full(num_trials, 1e6, dtype=np.float32)
    a = np.full(num_trials, 0.001, dtype=np.float32)
    tau = np.zeros(num_trials, dtype=np.float32)
    bias = np.zeros(num_trials, dtype=np.float32)

    data = sample_ddm(v, a, tau, bias)

    assert np.all(data["choice"] == 1.0)
    assert np.all(data["response_time"] > 0)


def test_sample_ddm_times_out_when_max_steps_too_small():
    num_trials = 5
    v = np.zeros(num_trials, dtype=np.float32)
    a = np.full(num_trials, 1e6, dtype=np.float32)
    tau = np.zeros(num_trials, dtype=np.float32)
    bias = np.full(num_trials, 0.5, dtype=np.float32)

    data = sample_ddm(v, a, tau, bias, max_steps=1)

    assert np.all(data["response_time"] == -1.0)
    assert np.all(data["choice"] == -1.0)
