import numpy as np
import pytest

from superstats.simulation.augmentation.missing_process import MissingProcess
from superstats.simulation.augmentation.random_missing import (
    DEFAULT_P_MISSING_PRIOR,
    RandomMissing,
)
from superstats.prior.prior import Prior


def test_missing_process_cannot_be_instantiated_directly():
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
    # Both entry points should work without an explicit rng, and both
    # should return the same dict contract.
    process = RandomMissing(p_missing=0.5, missing_value=-1)
    data = np.random.default_rng(1).normal(size=(4, 10, 2)).astype(np.float32)

    via_call = process(data)
    via_apply = process.apply(data)

    for result in (via_call, via_apply):
        assert set(result.keys()) == {"data", "missing_mask", "p_missing", "missing_value"}
        assert result["data"].shape == data.shape
        assert result["missing_mask"].shape == (4, 10)


def test_random_missing_shape_and_dtype():
    batch_size, num_steps, dim = 8, 20, 3
    data = np.random.default_rng(0).normal(size=(batch_size, num_steps, dim)).astype(np.float32)

    process = RandomMissing(p_missing=0.3, missing_value=-1)
    result = process.apply(data, rng=np.random.default_rng(42))

    assert result["data"].shape == data.shape
    assert result["missing_mask"].shape == (batch_size, num_steps)
    assert result["missing_mask"].dtype == np.bool
    assert np.all(np.isin(result["missing_mask"], [0, 1]))


def test_random_missing_whole_observation_is_dropped_together():
    # Whenever a time step is missing, every data dimension at that step
    # must carry missing_value -- not just some of them.
    batch_size, num_steps, dim = 5, 15, 4
    data = np.random.default_rng(1).normal(size=(batch_size, num_steps, dim)).astype(np.float32)
    missing_value = -1.0

    process = RandomMissing(p_missing=0.5, missing_value=missing_value)
    result = process.apply(data, rng=np.random.default_rng(7))

    mask = result["missing_mask"][:, :, 0].astype(bool)  # (batch, steps)
    filled = result["data"]

    assert np.all(filled[mask] == missing_value)
    # non-missing entries should be untouched
    assert np.array_equal(filled[~mask], data[~mask])


def test_random_missing_probability_zero_means_no_missing():
    data = np.random.default_rng(2).normal(size=(6, 12, 2)).astype(np.float32)
    process = RandomMissing(p_missing=0.0, missing_value=-1)
    result = process.apply(data, rng=np.random.default_rng(3))

    assert np.all(result["missing_mask"] == 0)
    assert np.array_equal(result["data"], data)


def test_random_missing_probability_one_means_all_missing():
    data = np.random.default_rng(4).normal(size=(6, 12, 2)).astype(np.float32)
    process = RandomMissing(p_missing=1.0, missing_value=-1)
    result = process.apply(data, rng=np.random.default_rng(5))

    assert np.all(result["missing_mask"] == 1)
    assert np.all(result["data"] == -1)


def test_random_missing_shared_across_batch_uses_one_mask_for_all():
    batch_size, num_steps, dim = 10, 30, 2
    data = np.random.default_rng(6).normal(size=(batch_size, num_steps, dim)).astype(np.float32)

    process = RandomMissing(p_missing=0.5, missing_value=-1, shared_across_batch=True)
    result = process.apply(data, rng=np.random.default_rng(8))

    mask = result["missing_mask"][:, :, 0]  # (batch, steps)
    # every row (dataset) in the batch should have the identical mask
    assert np.all(mask == mask[0])


def test_random_missing_independent_across_batch_gives_different_masks():
    batch_size, num_steps, dim = 20, 50, 2
    data = np.random.default_rng(9).normal(size=(batch_size, num_steps, dim)).astype(np.float32)

    process = RandomMissing(p_missing=0.5, missing_value=-1, shared_across_batch=False)
    result = process.apply(data, rng=np.random.default_rng(10))

    mask = result["missing_mask"][:, :, 0]
    # with independent per-dataset masks at p=0.5 over many steps, it's
    # overwhelmingly unlikely every row is identical to the first
    assert not np.all(mask == mask[0])


def test_random_missing_accepts_prior_for_p_missing():
    data = np.random.default_rng(11).normal(size=(5, 10, 2)).astype(np.float32)
    prior = Prior("beta", a=2.0, b=2.0)

    process = RandomMissing(p_missing=prior, missing_value=-1)
    result = process.apply(data, rng=np.random.default_rng(12))

    assert result["missing_mask"].shape == (5, 10)
    assert np.all(np.isin(result["missing_mask"], [0, 1]))


def test_random_missing_defaults_p_missing_to_default_prior_when_none():
    process = RandomMissing()
    assert process.p_missing is DEFAULT_P_MISSING_PRIOR
    assert isinstance(process.p_missing, Prior)


def test_random_missing_explicit_p_missing_is_kept_as_is():
    process = RandomMissing(p_missing=0.2)
    assert process.p_missing == 0.2


def test_random_missing_promotes_dtype_for_nan_fill_on_int_data():
    data = np.random.default_rng(13).integers(0, 100, size=(4, 8, 2)).astype(np.int32)
    process = RandomMissing(p_missing=1.0, missing_value=np.nan)
    result = process.apply(data, rng=np.random.default_rng(14))

    assert result["data"].dtype == np.float64
    assert np.all(np.isnan(result["data"]))


def test_random_missing_keeps_int_dtype_for_int_fill():
    data = np.random.default_rng(15).integers(0, 100, size=(4, 8, 2)).astype(np.int32)
    process = RandomMissing(p_missing=1.0, missing_value=-1)
    result = process.apply(data, rng=np.random.default_rng(16))

    assert np.issubdtype(result["data"].dtype, np.integer)
    assert np.all(result["data"] == -1)


def test_random_missing_per_dimension_missing_value():
    batch_size, num_steps, dim = 3, 6, 2
    data = np.random.default_rng(17).normal(size=(batch_size, num_steps, dim)).astype(np.float32)
    per_dim_fill = np.array([-1.0, -99.0], dtype=np.float32)

    process = RandomMissing(p_missing=1.0, missing_value=per_dim_fill)
    result = process.apply(data, rng=np.random.default_rng(18))

    assert np.all(result["data"][:, :, 0] == -1.0)
    assert np.all(result["data"][:, :, 1] == -99.0)


def test_random_missing_reproducible_with_seeded_rng():
    data = np.random.default_rng(19).normal(size=(6, 12, 2)).astype(np.float32)
    process = RandomMissing(p_missing=0.4, missing_value=-1)

    result_a = process.apply(data, rng=np.random.default_rng(100))
    result_b = process.apply(data, rng=np.random.default_rng(100))

    assert np.array_equal(result_a["missing_mask"], result_b["missing_mask"])
    assert np.array_equal(result_a["data"], result_b["data"])
