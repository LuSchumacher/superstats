import numpy as np

from superstats.diagnostics.metrics import (
    calibration_error_per_step,
    nrmse_per_step,
    posterior_contraction_per_step,
    correlation_per_step,
)

NUM_SIM = 20
NUM_SAMPLES = 30
NUM_STEPS = 5
NUM_PARAMS = 3


def _random_targets():
    return np.random.normal(size=(NUM_SIM, NUM_STEPS, NUM_PARAMS))


def _random_estimates_point():
    return np.random.normal(size=(NUM_SIM, NUM_STEPS, NUM_PARAMS))


def _random_estimates_posterior():
    return np.random.normal(size=(NUM_SIM, NUM_SAMPLES, NUM_STEPS, NUM_PARAMS))


def test_correlation_per_step_shape():
    targets = _random_targets()
    estimates = _random_estimates_point()

    correlation = correlation_per_step(estimates, targets)

    assert correlation.shape == (NUM_STEPS, NUM_PARAMS)
    assert np.all(np.isfinite(correlation))


def test_posterior_contraction_per_step_shape():
    targets = _random_targets()
    estimates = _random_estimates_posterior()

    contraction = posterior_contraction_per_step(estimates, targets)

    assert contraction.shape == (NUM_STEPS, NUM_PARAMS)
    assert np.all(np.isfinite(contraction))


def test_calibration_error_per_step_shape_and_bounds():
    targets = _random_targets()
    estimates = _random_estimates_posterior()

    calibration_error = calibration_error_per_step(estimates, targets, resolution=10)

    assert calibration_error.shape == (NUM_STEPS, NUM_PARAMS)
    assert np.all(calibration_error >= 0.0)
    assert np.all(calibration_error <= 1.0)


def test_nrmse_per_step_shape():
    targets = _random_targets()
    estimates = _random_estimates_posterior()

    nrmse = nrmse_per_step(estimates, targets)

    assert nrmse.shape == (NUM_STEPS, NUM_PARAMS)
    assert np.all(np.isfinite(nrmse))
