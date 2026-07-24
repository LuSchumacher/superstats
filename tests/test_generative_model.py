import logging

import numpy as np
import pandas as pd
import pytest

from superstats.prior import JointPrior, Prior
from superstats.simulation import GenerativeModel, sample_ddm
from superstats.simulation.augmentation import (
    ContaminationProcess,
    MissingProcess,
    RandomChoiceContamination,
    RandomMissingProcess,
)
from superstats.transition import DeterministicTransition, Linear
from superstats.transition.stochastic_transitions import RandomWalk
from superstats.workflow import Workflow

BATCH_SIZE = 4
NUM_STEPS = 6


class _DeterministicTestTransition(DeterministicTransition):
    def sample(self, batch_size, num_steps):
        trajectory = np.broadcast_to(np.arange(num_steps, dtype=np.float32), (batch_size, num_steps))
        return {"deterministic_params": trajectory, "hyper_params": {}, "fixed_params": {}}


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


def test_generative_model_rejects_non_dict_model_output():
    def array_model(v):
        return np.asarray(v)

    prior = JointPrior(v=Prior("normal", loc=0.0, scale=1.0))
    with pytest.raises(TypeError, match="model must return a dict"):
        GenerativeModel(prior=prior, model=array_model)


def test_generative_model_param_order_matches_model_signature():
    gm = _build_generative_model()
    assert gm.param_order == ["v", "a", "tau", "bias", "sigma", "dt", "max_steps"]
    assert gm.data_keys == ["response_time", "choice"]


def test_generative_model_sample_shapes():
    gm = _build_generative_model()
    result = gm.sample(batch_size=BATCH_SIZE, num_steps=NUM_STEPS)

    assert "data" not in result
    assert result["response_time"].shape == (BATCH_SIZE, NUM_STEPS)
    assert result["choice"].shape == (BATCH_SIZE, NUM_STEPS)
    assert result["v"].shape == (BATCH_SIZE, NUM_STEPS, 1)
    assert result["a"].shape == (BATCH_SIZE, 1)

    # fixed params are excluded by default
    assert "tau" not in result
    assert "bias" not in result
    assert np.all(np.isfinite(result["response_time"]))
    assert np.all(np.isin(result["choice"], [-1.0, 0.0, 1.0]))


def test_generative_model_sample_include_fixed():
    gm = _build_generative_model()
    result = gm.sample(batch_size=BATCH_SIZE, num_steps=NUM_STEPS, include_fixed=True)

    assert "tau" in result
    assert "bias" in result


def test_generative_model_sample_tile_to_steps():
    gm = _build_generative_model()
    result = gm.sample(batch_size=BATCH_SIZE, num_steps=NUM_STEPS, tile_to_steps=True)

    assert result["a"].shape == (BATCH_SIZE, NUM_STEPS, 1)
    np.testing.assert_allclose(result["a"], np.broadcast_to(result["a"][:, :1, :], result["a"].shape))


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

    assert set(sim_data) == {"response_time", "choice"}
    assert sim_data["response_time"].shape == (BATCH_SIZE, NUM_STEPS)
    assert sim_data["choice"].shape == (BATCH_SIZE, NUM_STEPS)


def test_generative_model_sample_includes_time_steps():
    gm = _build_generative_model()
    result = gm.sample(batch_size=BATCH_SIZE, num_steps=NUM_STEPS)

    assert "time_steps" in result
    assert result["time_steps"].shape == (BATCH_SIZE, NUM_STEPS)

    expected_row = np.arange(1, NUM_STEPS + 1)
    for row in result["time_steps"]:
        assert np.array_equal(row, expected_row)


def test_workflow_default_adapter_uses_named_time_series_data_keys():
    gm = _build_generative_model()
    adapter = Workflow.default_adapter(gm)
    result = gm.sample(batch_size=BATCH_SIZE, num_steps=NUM_STEPS, tile_to_steps=True)

    adapted = adapter(result)

    assert "data" not in adapted
    assert "response_time" not in adapted
    assert "choice" not in adapted
    assert adapted["summary_variables"].shape == (BATCH_SIZE, NUM_STEPS, 4)
    assert adapted["inference_variables"].shape == (BATCH_SIZE, NUM_STEPS, 2)


def test_deterministic_trajectories_are_simulated_but_not_inferred():
    prior = JointPrior(v=RandomWalk(sigma=0.05, delta=0.0), d=_DeterministicTestTransition())

    def model(v, d):
        return {"observation": v + d}

    gm = GenerativeModel(prior=prior, model=model, missing=None)
    result = gm.sample(batch_size=BATCH_SIZE, num_steps=NUM_STEPS, tile_to_steps=True)
    adapted = Workflow.default_adapter(gm)(result)

    assert gm.local_keys == ["v"]
    assert gm.deterministic_keys == ["d"]
    assert result["d"].shape == (BATCH_SIZE, NUM_STEPS, 1)
    assert adapted["inference_variables"].shape[-1] == 1


def test_resimulate_posterior_reconstructs_linear_deterministic_parameter():
    prior = JointPrior(
        v=RandomWalk(bounds=(-3.0, 3.0), initial_prior=Prior("normal", loc=0.0, scale=0.5), sigma=0.0, delta=0.0),
        a=Linear(intercept=0.5, beta=0.5),
        tau=0.2,
        bias=0.0,
    )
    simulator = GenerativeModel(prior=prior, model=sample_ddm, missing=None)
    workflow = object.__new__(Workflow)
    workflow.simulator = simulator

    posterior = {
        "v": np.zeros((2, 3, NUM_STEPS), dtype=np.float32),
        # Shared posterior outputs can be tiled over time by the adapter.
        "a_intercept": np.full((2, 3, NUM_STEPS, 1), 0.5, dtype=np.float32),
    }
    result = workflow.resimulate_posterior(posterior, num_sims=4, rng=0)

    assert result["response_time"].shape == (2, 4, NUM_STEPS)
    assert result["choice"].shape == (2, 4, NUM_STEPS)


def test_workflow_prepare_conditions_adds_time_steps_for_named_data(caplog):
    gm = _build_generative_model()
    workflow = object.__new__(Workflow)
    workflow.simulator = gm
    result = gm.sample(batch_size=BATCH_SIZE, num_steps=NUM_STEPS)

    with caplog.at_level(logging.WARNING, logger="superstats"):
        conditions = workflow._prepare_conditions({key: result[key] for key in gm.data_keys})

    assert set(conditions) == {"response_time", "choice", "time_steps", "missing_mask"}
    assert conditions["time_steps"].shape == (BATCH_SIZE, NUM_STEPS)
    assert np.array_equal(conditions["time_steps"][0], np.arange(1, NUM_STEPS + 1))
    assert not np.any(conditions["missing_mask"])
    assert "No time_steps provided" in caplog.text
    assert "No missing_mask provided" in caplog.text


def test_workflow_prepare_conditions_adds_default_mask_when_configured(caplog):
    gm = _build_generative_model(missing="random")
    workflow = object.__new__(Workflow)
    workflow.simulator = gm
    result = gm.sample(batch_size=BATCH_SIZE, num_steps=NUM_STEPS, rng=np.random.default_rng(0))

    with caplog.at_level(logging.WARNING, logger="superstats"):
        conditions = workflow._prepare_conditions({key: result[key] for key in gm.data_keys})

    assert set(conditions) == {"response_time", "choice", "time_steps", "missing_mask"}
    assert conditions["missing_mask"].shape == (BATCH_SIZE, NUM_STEPS)
    assert not np.any(conditions["missing_mask"])
    assert "No missing_mask provided" in caplog.text


def test_workflow_df_to_dict_normalizes_negative_discrete_time():
    gm = _build_generative_model()
    workflow = object.__new__(Workflow)
    workflow.simulator = gm
    df = pd.DataFrame(
        {
            "participant": ["a", "a", "a", "b", "b"],
            "time": [-2, 0, 2, -2, 2],
            "rt": [1.0, 2.0, 3.0, 4.0, 5.0],
            "choice": [0, 1, 0, 1, 1],
        }
    )

    data = workflow.df_to_dict(
        df,
        id_col="participant",
        time_col="time",
        data_mapping={"rt": "response_time", "choice": "choice"},
        missing_value=-1,
    )

    assert set(data) == {"response_time", "choice", "missing_mask", "time_steps"}
    assert np.array_equal(data["time_steps"], np.array([[1, 2, 3], [1, 2, 3]]))
    assert np.array_equal(data["response_time"], np.array([[1.0, 2.0, 3.0], [4.0, -1.0, 5.0]]))
    assert np.array_equal(data["missing_mask"], np.array([[False, False, False], [False, True, False]]))


def test_workflow_df_to_dict_rejects_continuous_time():
    gm = _build_generative_model()
    workflow = object.__new__(Workflow)
    workflow.simulator = gm
    df = pd.DataFrame(
        {
            "participant": ["a", "a"],
            "time": [0.0, 0.5],
            "rt": [1.0, 2.0],
            "choice": [0, 1],
        }
    )

    with pytest.raises(ValueError, match="discrete integer-like"):
        workflow.df_to_dict(
            df,
            id_col="participant",
            time_col="time",
            data_mapping={"rt": "response_time", "choice": "choice"},
        )


def test_workflow_df_to_dict_uses_simulator_missing_value():
    gm = _build_generative_model(missing=RandomMissingProcess(p_missing=0.0, missing_value=-999.0))
    workflow = object.__new__(Workflow)
    workflow.simulator = gm
    df = pd.DataFrame(
        {
            "participant": ["a", "a"],
            "rt": [1.0, -1.0],
            "choice": [0, 1],
        }
    )

    data = workflow.df_to_dict(
        df,
        id_col="participant",
        data_mapping={"rt": "response_time", "choice": "choice"},
        missing_value=-1,
    )

    assert np.array_equal(data["time_steps"], np.array([[1, 2]]))
    assert np.array_equal(data["missing_mask"], np.array([[False, True]]))
    assert data["response_time"][0, 1] == -999.0
    assert data["choice"][0, 1] == -999.0


def test_generative_model_defaults_to_random_missing():
    gm = _build_generative_model()
    assert isinstance(gm.missing, RandomMissingProcess)
    assert isinstance(gm.missing, MissingProcess)
    assert gm.has_mask is True


def test_generative_model_sample_includes_missing_mask_by_default():
    gm = _build_generative_model()
    result = gm.sample(batch_size=BATCH_SIZE, num_steps=NUM_STEPS, rng=np.random.default_rng(0))

    assert "missing_mask" in result
    assert result["missing_mask"].shape == (BATCH_SIZE, NUM_STEPS)
    assert result["missing_mask"].dtype == np.bool_
    assert "p_missing" in result


def test_generative_model_random_missing_includes_mask():
    gm = _build_generative_model(missing="random")
    result = gm.sample(batch_size=BATCH_SIZE, num_steps=NUM_STEPS, rng=np.random.default_rng(0))

    assert isinstance(gm.missing, RandomMissingProcess)
    assert isinstance(gm.missing, MissingProcess)
    assert gm.has_mask is True
    assert "missing_mask" in result
    assert result["missing_mask"].shape == (BATCH_SIZE, NUM_STEPS)
    assert result["missing_mask"].dtype == np.bool_


def test_generative_model_missing_none_disables_mask():
    gm = _build_generative_model(missing=None)
    assert gm.missing is None

    result = gm.sample(batch_size=BATCH_SIZE, num_steps=NUM_STEPS)
    assert "missing_mask" not in result


def test_generative_model_missing_none_leaves_data_unmodified():
    gm_no_missing = _build_generative_model(missing=None)
    gm_with_missing = _build_generative_model(missing=RandomMissingProcess(p_missing=1.0, missing_value=-999.0))

    rng = np.random.default_rng(1)
    result_none = gm_no_missing.sample(batch_size=BATCH_SIZE, num_steps=NUM_STEPS, rng=np.random.default_rng(1))
    result_all_missing = gm_with_missing.sample(batch_size=BATCH_SIZE, num_steps=NUM_STEPS, rng=rng)

    assert "data" not in result_none
    assert "data" not in result_all_missing
    assert not np.any(result_none["response_time"] == -999.0)
    assert not np.any(result_none["choice"] == -999.0)
    assert np.all(result_all_missing["response_time"] == -999.0)
    assert np.all(result_all_missing["choice"] == -999.0)


def test_generative_model_accepts_custom_missing_instance():
    custom = RandomMissingProcess(p_missing=0.0, missing_value=-1.0, shared_across_batch=True)
    gm = _build_generative_model(missing=custom)

    assert gm.missing is custom

    result = gm.sample(batch_size=BATCH_SIZE, num_steps=NUM_STEPS, rng=np.random.default_rng(2))
    # p_missing=0.0 -> nothing should be marked missing
    assert np.all(result["missing_mask"] == 0)


def test_generative_model_missing_receives_rng_for_reproducibility():
    # Note: `self.prior.sample(...)` in `GenerativeModel.sample` is not
    # itself seeded by `rng` (it uses global `np.random` state internally),
    # so only the missingness draw -- not the underlying simulated data --
    # is guaranteed reproducible here. We isolate that by masking the same
    # fixed array twice rather than asserting on `sample()`'s full output.
    process = RandomMissingProcess(p_missing=0.5)
    rng = np.random.default_rng(0)
    data_a = {
        "response_time": rng.normal(size=(BATCH_SIZE, NUM_STEPS)).astype(np.float32),
        "choice": rng.integers(0, 2, size=(BATCH_SIZE, NUM_STEPS)).astype(np.float32),
    }
    data_b = {key: value.copy() for key, value in data_a.items()}

    result_a = process.apply(data_a, rng=np.random.default_rng(42))
    result_b = process.apply(data_b, rng=np.random.default_rng(42))

    assert np.array_equal(result_a["missing_mask"], result_b["missing_mask"])
    for key in data_a:
        assert np.array_equal(result_a[key], result_b[key])


def test_generative_model_accepts_plain_callable_missing():
    def half_missing(data, rng=None):
        example = next(iter(data.values()))
        mask = np.zeros(example.shape, dtype=bool)
        mask[:, : example.shape[1] // 2] = True
        filled = {key: value.copy() for key, value in data.items()}
        for value in filled.values():
            value[mask] = -1.0
        return filled | {"missing_mask": mask}

    gm = _build_generative_model(missing=half_missing)
    result = gm.sample(batch_size=BATCH_SIZE, num_steps=NUM_STEPS)

    assert np.all(result["missing_mask"][:, : NUM_STEPS // 2])
    assert not np.any(result["missing_mask"][:, NUM_STEPS // 2 :])
    assert "data" not in result
    assert np.all(result["response_time"][:, : NUM_STEPS // 2] == -1.0)
    assert np.all(result["choice"][:, : NUM_STEPS // 2] == -1.0)


def test_generative_model_rejects_non_callable_missing():
    with pytest.raises(TypeError):
        _build_generative_model(missing="not-callable")


def test_generative_model_random_choice_contamination():
    gm = _build_generative_model(contamination="random_choice")

    assert isinstance(gm.contamination, RandomChoiceContamination)
    assert isinstance(gm.contamination, ContaminationProcess)


def test_generative_model_rejects_non_callable_contamination():
    with pytest.raises(TypeError):
        _build_generative_model(contamination="not-callable")


def test_generative_model_propagates_missing_contract_errors():
    def bad_process(data, rng=None):
        return data

    gm = _build_generative_model(missing=bad_process)
    with pytest.raises(KeyError):
        gm.sample(batch_size=BATCH_SIZE, num_steps=NUM_STEPS)
