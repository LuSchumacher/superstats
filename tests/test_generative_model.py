import numpy as np
import pytest

from superstats.prior import JointPrior, Prior
from superstats.simulation import GenerativeModel, sample_ddm
from superstats.simulation.augmentation import MissingProcess, RandomMissing
from superstats.transition import RandomWalk

BATCH_SIZE = 4
NUM_STEPS = 6


def _build_generative_model(**kwargs):
    prior = JointPrior(
        v=RandomWalk(bounds=(-3.0, 3.0), initial_prior=Prior("normal", loc=0.0, scale=0.5), sigma=0.05, delta=0.0),
        a=Prior("halfnormal", scale=1.0),
        tau=0.2,
        bias=0.0,
    )
    return GenerativeModel(prior=prior, model=sample_ddm, **kwargs)


def test_generative_model_rejects_non_callable_model():
    prior = JointPrior(a=Prior("halfnormal", scale=1.0))
    with pytest.raises(TypeError):
        GenerativeModel(prior=prior, model="not-callable")


def test_generative_model_param_order_matches_model_signature():
    gm = _build_generative_model()
    assert gm.param_order == ["v", "a", "tau", "bias", "sigma", "dt", "max_steps"]


def test_generative_model_sample_shapes():
    gm = _build_generative_model()
    result = gm.sample(batch_size=BATCH_SIZE, num_steps=NUM_STEPS)

    assert result["data"].shape == (BATCH_SIZE, NUM_STEPS, 2)
    assert result["v"].shape == (BATCH_SIZE, NUM_STEPS, 1)
    assert result["a"].shape == (BATCH_SIZE, 1)

    # fixed params are excluded by default
    assert "tau" not in result
    assert "bias" not in result


def test_generative_model_sample_include_fixed():
    gm = _build_generative_model()
    result = gm.sample(batch_size=BATCH_SIZE, num_steps=NUM_STEPS, include_fixed=True)

    assert "tau" in result
    assert "bias" in result


def test_generative_model_sample_tile_to_steps():
    gm = _build_generative_model()
    result = gm.sample(batch_size=BATCH_SIZE, num_steps=NUM_STEPS, tile_to_steps=True)

    assert result["a"].shape == (BATCH_SIZE, NUM_STEPS, 1)


def test_generative_model_get_fixed_params():
    gm = _build_generative_model()
    fixed_params = gm.get_fixed_params()

    assert fixed_params["tau"] == pytest.approx(0.2)
    assert fixed_params["bias"] == pytest.approx(0.0)


def test_generative_model_simulate_from_parameters():
    gm = _build_generative_model()
    params = {
        "v": np.zeros((BATCH_SIZE, NUM_STEPS), dtype=np.float32),
        "a": np.full(BATCH_SIZE, 1.0, dtype=np.float32),
        "tau": 0.2,
        "bias": 0.0,
    }

    sim_data = gm.simulate_from_parameters(params, batch_size=BATCH_SIZE, num_steps=NUM_STEPS)

    assert sim_data.shape == (BATCH_SIZE, NUM_STEPS, 2)


def test_generative_model_sample_includes_time_steps():
    gm = _build_generative_model()
    result = gm.sample(batch_size=BATCH_SIZE, num_steps=NUM_STEPS)

    assert "time_steps" in result
    assert result["time_steps"].shape == (BATCH_SIZE, NUM_STEPS)

    expected_row = np.arange(NUM_STEPS)
    for row in result["time_steps"]:
        assert np.array_equal(row, expected_row)


def test_generative_model_defaults_to_no_missing_process():
    gm = _build_generative_model()
    assert gm.missing_process is None
    assert gm.has_mask is False


def test_generative_model_sample_omits_missing_mask_by_default():
    gm = _build_generative_model()
    result = gm.sample(batch_size=BATCH_SIZE, num_steps=NUM_STEPS, rng=np.random.default_rng(0))

    assert "missing_mask" not in result


def test_generative_model_random_missing_process_includes_mask():
    gm = _build_generative_model(missing_process="random")
    result = gm.sample(batch_size=BATCH_SIZE, num_steps=NUM_STEPS, rng=np.random.default_rng(0))

    assert isinstance(gm.missing_process, RandomMissing)
    assert isinstance(gm.missing_process, MissingProcess)
    assert gm.has_mask is True
    assert "missing_mask" in result
    assert result["missing_mask"].shape == (BATCH_SIZE, NUM_STEPS)
    assert result["missing_mask"].dtype == np.bool_


def test_generative_model_missing_process_none_disables_mask():
    gm = _build_generative_model(missing_process=None)
    assert gm.missing_process is None

    result = gm.sample(batch_size=BATCH_SIZE, num_steps=NUM_STEPS)
    assert "missing_mask" not in result


def test_generative_model_missing_process_none_leaves_data_unmodified():
    gm_no_missing = _build_generative_model(missing_process=None)
    gm_with_missing = _build_generative_model(missing_process=RandomMissing(p_missing=1.0, missing_value=-999.0))

    rng = np.random.default_rng(1)
    result_none = gm_no_missing.sample(batch_size=BATCH_SIZE, num_steps=NUM_STEPS, rng=np.random.default_rng(1))
    result_all_missing = gm_with_missing.sample(batch_size=BATCH_SIZE, num_steps=NUM_STEPS, rng=rng)

    assert not np.any(result_none["data"] == -999.0)
    assert np.all(result_all_missing["data"] == -999.0)


def test_generative_model_accepts_custom_missing_process_instance():
    custom = RandomMissing(p_missing=0.0, missing_value=-1.0, shared_across_batch=True)
    gm = _build_generative_model(missing_process=custom)

    assert gm.missing_process is custom

    result = gm.sample(batch_size=BATCH_SIZE, num_steps=NUM_STEPS, rng=np.random.default_rng(2))
    # p_missing=0.0 -> nothing should be marked missing
    assert np.all(result["missing_mask"] == 0)


def test_generative_model_missing_process_receives_rng_for_reproducibility():
    # Note: `self.prior.sample(...)` in `GenerativeModel.sample` is not
    # itself seeded by `rng` (it uses global `np.random` state internally),
    # so only the missingness draw -- not the underlying simulated data --
    # is guaranteed reproducible here. We isolate that by masking the same
    # fixed array twice rather than asserting on `sample()`'s full output.
    process = RandomMissing(p_missing=0.5)
    data = np.random.default_rng(0).normal(size=(BATCH_SIZE, NUM_STEPS, 2)).astype(np.float32)

    result_a = process.apply(data, rng=np.random.default_rng(42))
    result_b = process.apply(data, rng=np.random.default_rng(42))

    assert np.array_equal(result_a["missing_mask"], result_b["missing_mask"])
    assert np.array_equal(result_a["data"], result_b["data"])


def test_generative_model_accepts_plain_callable_missing_process():
    def half_missing(data, rng=None):
        mask = np.zeros(data.shape[:2], dtype=bool)
        mask[:, : data.shape[1] // 2] = True
        filled = data.copy()
        filled[:, : data.shape[1] // 2, :] = -1.0
        return {"data": filled, "missing_mask": mask}

    gm = _build_generative_model(missing_process=half_missing)
    result = gm.sample(batch_size=BATCH_SIZE, num_steps=NUM_STEPS)

    assert np.all(result["missing_mask"][:, : NUM_STEPS // 2])
    assert not np.any(result["missing_mask"][:, NUM_STEPS // 2 :])
    assert np.all(result["data"][:, : NUM_STEPS // 2, :] == -1.0)


def test_generative_model_rejects_non_callable_missing_process():
    with pytest.raises(TypeError):
        _build_generative_model(missing_process="not-callable")


def test_generative_model_rejects_missing_process_with_bad_return_contract():
    def bad_process(data, rng=None):
        return data  # not a dict with 'data'/'missing_mask'

    gm = _build_generative_model(missing_process=bad_process)
    with pytest.raises(TypeError):
        gm.sample(batch_size=BATCH_SIZE, num_steps=NUM_STEPS)
