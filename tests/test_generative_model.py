import numpy as np
import pytest

from superstats.prior import JointPrior, Prior
from superstats.simulation import GenerativeModel, sample_ddm
from superstats.transition import RandomWalk

BATCH_SIZE = 4
NUM_STEPS = 6


def _build_generative_model():
    prior = JointPrior(
        v=RandomWalk(bounds=(-3.0, 3.0), initial_prior=Prior("normal", loc=0.0, scale=0.5), sigma=0.05, delta=0.0),
        a=Prior("halfnormal", scale=1.0),
        tau=0.2,
        bias=0.0,
    )
    return GenerativeModel(prior=prior, model=sample_ddm)


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
