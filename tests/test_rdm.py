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

    assert isinstance(data, np.ndarray)
    assert data.shape == (num_trials, 2)
    assert data.dtype == np.float32

    rt, choice = data[:, 0], data[:, 1]
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
    assert np.all(data[:, 1] == 1.0)
    assert np.all(data[:, 0] > 0)


def test_sample_rdm_times_out_when_max_steps_too_small():
    num_trials = 5
    v_base = np.zeros(num_trials, dtype=np.float32)
    v_diff = np.zeros(num_trials, dtype=np.float32)
    a_base = np.full(num_trials, 1e6, dtype=np.float32)
    tau = np.zeros(num_trials, dtype=np.float32)
    bias = np.full(num_trials, 1.0, dtype=np.float32)
    sigma_diff = np.full(num_trials, 1.0, dtype=np.float32)

    data = sample_rdm(v_base, v_diff, a_base, tau, bias, sigma_diff, max_steps=1)

    assert np.all(data[:, 0] == -1.0)
    assert np.all(data[:, 1] == -1.0)
