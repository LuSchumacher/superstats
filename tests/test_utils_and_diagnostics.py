import numpy as np
import pytest

from superstats.diagnostics.metrics import (
    calibration_error_per_step,
    correlation_per_step,
    nrmse_per_step,
    posterior_contraction_per_step,
)
from superstats.networks.utils import expand_singletons_to_common_length
from superstats.utils.plotting import prepare_plot_data


def test_expand_singletons_broadcasts_scalars_and_singletons():
    result = expand_singletons_to_common_length(width=16, activation=["relu", "tanh"], bias=[True])

    assert result == {"width": [16, 16], "activation": ["relu", "tanh"], "bias": [True, True]}


@pytest.mark.parametrize("kwargs", [{"a": []}, {"a": [1, 2], "b": [3, 4, 5]}])
def test_expand_singletons_rejects_empty_or_incompatible_sequences(kwargs):
    with pytest.raises(ValueError):
        expand_singletons_to_common_length(**kwargs)


def test_prepare_plot_data_selects_and_stacks_named_variables():
    estimates = {"b": np.ones((2, 3, 4)), "a": np.zeros((2, 3, 4))}
    targets = {"b": np.ones((2, 4)), "a": np.zeros((2, 4))}

    est, target, names = prepare_plot_data(estimates, targets, variable_keys=["a", "b"], variable_names=["A", "B"])

    assert names == ["A", "B"]
    assert est.shape == (2, 3, 4, 2)
    assert target.shape == (2, 4, 2)
    np.testing.assert_array_equal(est[..., 0], 0)
    np.testing.assert_array_equal(target[..., 1], 1)


def test_prepare_plot_data_rejects_mixed_inputs_and_unknown_names():
    with pytest.raises(ValueError, match="must both"):
        prepare_plot_data({"x": np.ones(2)}, np.ones((2, 1)))
    with pytest.raises(ValueError, match="not found"):
        prepare_plot_data({"x": np.ones(2)}, {"x": np.ones(2)}, variable_keys=["missing"])


def test_diagnostics_perfect_posterior_has_expected_invariants():
    targets = np.array([[[0.0], [1.0]], [[2.0], [3.0]], [[4.0], [5.0]]])
    estimates = np.repeat(targets[:, None], 4, axis=1)

    np.testing.assert_allclose(correlation_per_step(estimates, targets), 1.0)
    np.testing.assert_allclose(posterior_contraction_per_step(estimates, targets), 1.0)
    np.testing.assert_allclose(nrmse_per_step(estimates, targets), 0.0)
    calibration = calibration_error_per_step(estimates, targets)
    assert calibration.shape == (2, 1)
    assert np.all((calibration >= 0) & (calibration <= 1))


@pytest.mark.parametrize("fn", [correlation_per_step, posterior_contraction_per_step, nrmse_per_step])
def test_diagnostics_reject_non_four_dimensional_estimates(fn):
    with pytest.raises(ValueError, match="shape"):
        fn(np.ones((2, 3, 4)), np.ones((2, 4, 1)))


def test_calibration_rejects_invalid_quantiles():
    with pytest.raises(ValueError, match="Require"):
        calibration_error_per_step(np.ones((2, 3, 1, 1)), np.ones((2, 1, 1)), min_quantile=0.5, max_quantile=0.5)
