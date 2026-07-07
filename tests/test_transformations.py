import numpy as np

from superstats.utils.transformations import scaled_sigmoid


def test_scaled_sigmoid_scalar_midpoint():
    # x=0 maps to the midpoint of [lower, upper]
    result = scaled_sigmoid(0.0, -2.0, 2.0)
    assert np.isclose(result, 0.0)


def test_scaled_sigmoid_scalar_bounds():
    lower, upper = -3.0, 5.0
    low_result = scaled_sigmoid(-50.0, lower, upper)
    high_result = scaled_sigmoid(50.0, lower, upper)

    assert lower <= low_result <= upper
    assert lower <= high_result <= upper
    assert np.isclose(low_result, lower, atol=1e-3)
    assert np.isclose(high_result, upper, atol=1e-3)


def test_scaled_sigmoid_array_shape_and_bounds():
    lower, upper = 0.0, 1.0
    x = np.linspace(-10, 10, 50)
    result = scaled_sigmoid(x, lower, upper)

    assert result.shape == x.shape
    assert np.all(result >= lower)
    assert np.all(result <= upper)
