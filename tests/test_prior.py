import numpy as np
import pytest

from superstats.prior import Prior


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dist": "normal", "loc": 0.0, "scale": 1.0},
        {"dist": "halfnormal", "scale": 1.0},
        {"dist": "uniform", "low": 0.0, "high": 1.0},
        {"dist": "beta", "a": 2.0, "b": 3.0},
        {"dist": "logistic", "loc": 0.0, "scale": 1.0},
    ],
)
def test_prior_sample_shape_and_dtype(kwargs):
    prior = Prior(**kwargs)
    samples = prior.sample(batch_size=32)

    assert isinstance(samples, np.ndarray)
    assert samples.shape == (32,)
    assert samples.dtype == np.float32


def test_prior_dirichlet_sample_shape():
    prior = Prior(dist="dirichlet", alpha=[1.0, 1.0, 1.0])
    samples = prior.sample(batch_size=16)

    assert samples.shape == (16, 3)
    assert samples.dtype == np.float32
    # each row of a dirichlet sample lies on the simplex
    np.testing.assert_allclose(samples.sum(axis=1), 1.0, rtol=1e-5, atol=1e-5)


def test_prior_dirichlet_requires_alpha():
    prior = Prior(dist="dirichlet")
    with pytest.raises(ValueError):
        prior.sample(batch_size=4)


def test_prior_unsupported_distribution_raises():
    prior = Prior(dist="not-a-distribution")
    with pytest.raises(ValueError):
        prior.sample(batch_size=4)


def test_prior_halfnormal_is_nonnegative():
    prior = Prior(dist="halfnormal", scale=1.0)
    samples = prior.sample(batch_size=100)
    assert np.all(samples >= 0.0)
