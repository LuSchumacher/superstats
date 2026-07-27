import numpy as np
import pytest

from superstats.prior.prior import Prior
from superstats.simulation.augmentation.missing import MissingProcess
from superstats.simulation.augmentation.random_missing import (
    DEFAULT_P_MISSING_PRIOR,
    RandomMissingProcess,
)


def _named_data(batch_size=4, num_steps=10, dtype=np.float32):
    rng = np.random.default_rng(123)
    return {
        "response_time": rng.normal(size=(batch_size, num_steps)).astype(dtype),
        "choice": rng.integers(0, 2, size=(batch_size, num_steps)).astype(dtype),
    }


def test_missing_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        MissingProcess()


def test_default_rng_returns_generator_when_none():
    rng = MissingProcess._default_rng(None)
    assert isinstance(rng, np.random.Generator)


def test_default_rng_passes_through_given_generator():
    given = np.random.default_rng(0)
    rng = MissingProcess._default_rng(given)
    assert rng is given


def test_call_and_apply_are_interchangeable_with_no_rng():
    process = RandomMissingProcess(p_missing=0.5, missing_value=-1)
    data = _named_data(batch_size=4, num_steps=10)

    via_call = process(data)
    via_apply = process.apply(data)

    for result in (via_call, via_apply):
        assert set(result) == {"response_time", "choice", "missing_mask", "p_missing"}
        assert result["response_time"].shape == (4, 10)
        assert result["choice"].shape == (4, 10)
        assert result["missing_mask"].shape == (4, 10)


def test_random_missing_shape_and_dtype():
    batch_size, num_steps = 8, 20
    data = _named_data(batch_size=batch_size, num_steps=num_steps)

    process = RandomMissingProcess(p_missing=0.3, missing_value=-1)
    result = process.apply(data, rng=np.random.default_rng(42))

    assert result["response_time"].shape == (batch_size, num_steps)
    assert result["choice"].shape == (batch_size, num_steps)
    assert result["missing_mask"].shape == (batch_size, num_steps)
    assert result["missing_mask"].dtype == np.bool_
    assert np.all(np.isin(result["missing_mask"], [0, 1]))


def test_random_missing_accepts_per_variable_missing_values():
    batch_size, num_steps = 4, 8
    data = _named_data(batch_size=batch_size, num_steps=num_steps)
    missing_value = np.array([-1.0, -99.0], dtype=np.float32)

    process = RandomMissingProcess(p_missing=1.0, missing_value=missing_value)
    result = process.apply(data, rng=np.random.default_rng(42))

    assert {"response_time", "choice"}.issubset(result)
    assert result["missing_mask"].shape == (batch_size, num_steps)
    assert np.all(result["response_time"] == -1.0)
    assert np.all(result["choice"] == -99.0)


def test_random_missing_accepts_per_key_missing_values():
    data = _named_data(batch_size=3, num_steps=6)
    missing_value = {"response_time": -1.0, "choice": -99.0}

    process = RandomMissingProcess(p_missing=1.0, missing_value=missing_value)
    result = process.apply(data, rng=np.random.default_rng(18))

    assert np.all(result["response_time"] == -1.0)
    assert np.all(result["choice"] == -99.0)


def test_random_missing_whole_observation_is_dropped_together():
    data = _named_data(batch_size=5, num_steps=15)
    original = {key: value.copy() for key, value in data.items()}
    missing_value = -1.0

    process = RandomMissingProcess(p_missing=0.5, missing_value=missing_value)
    result = process.apply(data, rng=np.random.default_rng(7))

    mask = result["missing_mask"].astype(bool)
    for key in data:
        filled = result[key]
        assert np.all(filled[mask] == missing_value)
        assert np.array_equal(filled[~mask], original[key][~mask])


def test_random_missing_probability_zero_means_no_missing():
    data = _named_data(batch_size=6, num_steps=12)
    process = RandomMissingProcess(p_missing=0.0, missing_value=-1)
    result = process.apply(data, rng=np.random.default_rng(3))

    assert np.all(result["missing_mask"] == 0)
    for key in data:
        assert np.array_equal(result[key], data[key])


def test_random_missing_probability_one_means_all_missing():
    data = _named_data(batch_size=6, num_steps=12)
    process = RandomMissingProcess(p_missing=1.0, missing_value=-1)
    result = process.apply(data, rng=np.random.default_rng(5))

    assert np.all(result["missing_mask"] == 1)
    for key in data:
        assert np.all(result[key] == -1)


def test_random_missing_shared_across_batch_uses_one_mask_for_all():
    data = _named_data(batch_size=10, num_steps=30)

    process = RandomMissingProcess(p_missing=0.5, missing_value=-1, shared_across_batch=True)
    result = process.apply(data, rng=np.random.default_rng(8))

    mask = result["missing_mask"]
    assert np.all(mask == mask[0])


def test_random_missing_independent_across_batch_gives_different_masks():
    data = _named_data(batch_size=20, num_steps=50)

    process = RandomMissingProcess(p_missing=0.5, missing_value=-1, shared_across_batch=False)
    result = process.apply(data, rng=np.random.default_rng(10))

    mask = result["missing_mask"]
    assert not np.all(mask == mask[0])


def test_random_missing_accepts_prior_for_p_missing():
    data = _named_data(batch_size=5, num_steps=10)
    prior = Prior("beta", a=2.0, b=2.0)

    process = RandomMissingProcess(p_missing=prior, missing_value=-1)
    result = process.apply(data, rng=np.random.default_rng(12))

    assert result["missing_mask"].shape == (5, 10)
    assert np.all(np.isin(result["missing_mask"], [0, 1]))


def test_random_missing_defaults_p_missing_to_default_prior_when_none():
    process = RandomMissingProcess()
    assert process.p_missing is DEFAULT_P_MISSING_PRIOR
    assert isinstance(process.p_missing, Prior)


def test_random_missing_explicit_p_missing_is_kept_as_is():
    process = RandomMissingProcess(p_missing=0.2)
    assert process.p_missing == 0.2


def test_random_missing_promotes_dtype_for_nan_fill_on_int_data():
    data = _named_data(batch_size=4, num_steps=8, dtype=np.int32)
    process = RandomMissingProcess(p_missing=1.0, missing_value=np.nan)
    result = process.apply(data, rng=np.random.default_rng(14))

    for key in data:
        value = result[key]
        assert value.dtype == np.float64
        assert np.all(np.isnan(value))


def test_random_missing_keeps_int_dtype_for_int_fill():
    data = _named_data(batch_size=4, num_steps=8, dtype=np.int32)
    process = RandomMissingProcess(p_missing=1.0, missing_value=-1)
    result = process.apply(data, rng=np.random.default_rng(16))

    for key in data:
        value = result[key]
        assert np.issubdtype(value.dtype, np.integer)
        assert np.all(value == -1)


def test_random_missing_reproducible_with_seeded_rng():
    data_a = _named_data(batch_size=6, num_steps=12)
    data_b = {key: value.copy() for key, value in data_a.items()}
    process = RandomMissingProcess(p_missing=0.4, missing_value=-1)

    result_a = process.apply(data_a, rng=np.random.default_rng(100))
    result_b = process.apply(data_b, rng=np.random.default_rng(100))

    assert np.array_equal(result_a["missing_mask"], result_b["missing_mask"])
    for key in data_a:
        assert np.array_equal(result_a[key], result_b[key])


def test_random_missing_does_not_mutate_input_arrays():
    data = _named_data(batch_size=3, num_steps=5)
    original = {key: value.copy() for key, value in data.items()}

    RandomMissingProcess(p_missing=0.5, missing_value=-1).apply(data, rng=np.random.default_rng(101))

    for key in data:
        assert np.array_equal(data[key], original[key])
