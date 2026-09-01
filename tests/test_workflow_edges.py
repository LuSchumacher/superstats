import pickle
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from superstats.workflow import Workflow
from superstats.workflow import workflow as workflow_module


def bare_workflow(model=None):
    workflow = object.__new__(Workflow)
    workflow.model = model
    return workflow


def time_varying_data(keys=("theta", "eta")):
    targets = {key: np.arange(6, dtype=float).reshape(2, 3, 1) for key in keys}
    estimates = {key: np.arange(24, dtype=float).reshape(2, 4, 3, 1) for key in keys}
    return targets, estimates


def test_checkpoint_warning_filter_only_suppresses_matching_message():
    warning_filter = workflow_module._SuppressCheckpointExistsWarning()
    assert warning_filter.filter(SimpleNamespace(getMessage=lambda: "another warning"))
    assert not warning_filter.filter(SimpleNamespace(getMessage=lambda: "Checkpoint file exists already"))


def test_init_restores_checkpoint_and_history(monkeypatch, tmp_path):
    restored_model = object()
    restored_history = SimpleNamespace(history={"loss": [1.0]})
    with (tmp_path / "history.pkl").open("wb") as file:
        pickle.dump(restored_history, file)

    class FakeBasicWorkflow:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.approximator = None
            self.history = None

    load_model = Mock(return_value=restored_model)
    monkeypatch.setattr(workflow_module.bf, "BasicWorkflow", FakeBasicWorkflow)
    monkeypatch.setattr(workflow_module.keras.saving, "load_model", load_model)
    monkeypatch.setattr(workflow_module, "find_embedding_network", Mock(side_effect=lambda value: value))
    monkeypatch.setattr(workflow_module, "find_inference_network", Mock(side_effect=lambda value: value))

    workflow = Workflow(
        adapter=object(),
        embedding_network=object(),
        inference_network=object(),
        checkpoint_filepath=str(tmp_path),
    )

    load_model.assert_called_once_with(str(tmp_path / "model.keras"))
    assert workflow.approximator is restored_model
    assert workflow.history.history == {"loss": [1.0]}


def test_history_load_and_save_noops_and_merges_runs(tmp_path):
    workflow = bare_workflow()
    workflow.workflow = SimpleNamespace(history=SimpleNamespace(history={"loss": [3.0]}))
    workflow.checkpoint_filepath = None
    workflow._load_history()
    workflow._save_history(SimpleNamespace(history={"loss": [2.0]}))
    assert workflow.history.history == {"loss": [3.0]}

    workflow.checkpoint_filepath = str(tmp_path)
    new_history = SimpleNamespace(history={"loss": [2.0], "val_loss": [4.0]})
    workflow._save_history(new_history)

    assert workflow.history.history == {"loss": [3.0, 2.0], "val_loss": [4.0]}
    with (tmp_path / "history.pkl").open("rb") as file:
        persisted = pickle.load(file)
    assert persisted.history == workflow.history.history


def test_prepare_conditions_validates_and_filters_named_arrays():
    workflow = bare_workflow(SimpleNamespace(data_keys=["x"], has_mask=True))
    x = np.ones((2, 3))

    with pytest.raises(TypeError, match="must be a mapping"):
        workflow._prepare_conditions(x)
    with pytest.raises(KeyError, match="Missing observed data"):
        workflow._prepare_conditions({"other": x})
    with pytest.raises(ValueError, match="time_steps.*shape"):
        workflow._prepare_conditions({"x": x, "time_steps": np.ones((2, 2))})
    with pytest.raises(ValueError, match="missing_mask.*shape"):
        workflow._prepare_conditions({"x": x, "time_steps": np.ones((2, 3)), "missing_mask": np.ones((2, 2))})

    conditions = workflow._prepare_conditions({"x": x, "unused": x})
    assert set(conditions) == {"x", "time_steps", "missing_mask"}

    workflow.model = None
    source = {"anything": x}
    assert workflow._prepare_conditions(source) == source


def test_sample_prepares_conditions_and_forwards_sampling_options():
    workflow = bare_workflow()
    workflow._prepare_conditions = Mock(return_value={"prepared": np.ones(1)})
    workflow.workflow = SimpleNamespace(approximator=Mock())
    workflow.approximator.sample.return_value = {"theta": np.ones(1)}

    result = workflow.sample({"x": np.ones(1)}, num_samples=12, batch_size=3, seed=7)

    assert set(result) == {"theta"}
    workflow.approximator.sample.assert_called_once_with(
        conditions={"prepared": np.ones(1)}, num_samples=12, batch_size=3, seed=7
    )


@pytest.mark.parametrize(
    ("estimates", "match"),
    [
        ({}, "non-empty"),
        ({"theta": np.ones((2, 3))}, "at least 3 dimensions"),
        (
            {"theta": np.ones((2, 3, 1)), "eta": np.ones((3, 3, 1))},
            "batch size 3 but expected 2",
        ),
        ({"theta": np.ones((2, 3, 1, 1, 1))}, "Unexpected posterior shape"),
    ],
)
def test_resimulate_rejects_invalid_posterior_shapes(estimates, match):
    model = SimpleNamespace(local_keys=[], deterministic_keys=[], get_fixed_params=lambda: {})
    workflow = bare_workflow(model)

    with pytest.raises(ValueError, match=match):
        workflow.resimulate(estimates, num_sims=2, rng=0)


def test_plot_history_applies_defaults_and_tick_fontsize(monkeypatch):
    axes = [Mock(), Mock()]
    figure = SimpleNamespace(axes=axes)
    loss = Mock(return_value=figure)
    monkeypatch.setattr(workflow_module.bf.diagnostics.plots, "loss", loss)

    history = object()
    result = bare_workflow().plot_history(history, tick_fontsize=13, train_color="red")

    assert result is figure
    assert loss.call_args.args == (history,)
    assert loss.call_args.kwargs["train_color"] == "red"
    assert loss.call_args.kwargs["legend_fontsize"] == workflow_module.LABEL_FONTSIZE
    for axis in axes:
        axis.tick_params.assert_called_once_with(labelsize=13)


def test_verify_time_varying_squeezes_selected_variables(monkeypatch):
    plot = Mock(return_value="figure")
    monkeypatch.setattr(workflow_module, "plot_time_varying_verification", plot)
    targets, estimates = time_varying_data()
    workflow = bare_workflow(SimpleNamespace(local_keys=["theta", "eta"]))

    result = workflow.verify_time_varying(targets, estimates, variable_keys=["eta"], color="red")

    assert result == "figure"
    assert plot.call_args.kwargs["targets"]["eta"].shape == (2, 3)
    assert plot.call_args.kwargs["estimates"]["eta"].shape == (2, 4, 3)
    assert plot.call_args.kwargs["color"] == "red"


def test_prepare_time_varying_at_steps_selects_and_labels_columns():
    targets, estimates = time_varying_data()
    workflow = bare_workflow(SimpleNamespace(local_keys=["theta", "eta"]))

    estimate_array, target_array, names = workflow._prepare_time_varying_at_steps(
        targets,
        estimates,
        time_steps=[0, -1],
        variable_names=["Theta", "Eta"],
    )

    assert estimate_array.shape == (2, 4, 4)
    assert target_array.shape == (2, 4)
    assert names == ["Theta (step 0)", "Eta (step 0)", "Theta (step 2)", "Eta (step 2)"]
    np.testing.assert_array_equal(target_array[:, 0], targets["theta"][:, 0, 0])
    np.testing.assert_array_equal(estimate_array[:, :, -1], estimates["eta"][:, :, -1, 0])


@pytest.mark.parametrize(
    ("mutate", "time_steps", "variable_keys", "variable_names", "error", "match"),
    [
        (None, 0, [], None, ValueError, "No time-varying"),
        (None, 0, ["missing"], None, ValueError, "not found"),
        (None, 0, ["theta"], [], ValueError, "variable_names"),
        ("bad_target", 0, ["theta"], None, ValueError, "Target 'theta'"),
        ("bad_estimate", 0, ["theta"], None, ValueError, "Estimate 'theta'"),
        ("inconsistent_pair", 0, ["theta"], None, ValueError, "inconsistent"),
        ("inconsistent_variables", 0, None, None, ValueError, "matching dataset"),
        (None, [], None, None, ValueError, "at least one"),
        (None, [0, 1.5], None, None, TypeError, "sequence of ints"),
        (None, True, None, None, TypeError, "sequence of ints"),
        (None, "0", None, None, TypeError, "sequence of ints"),
        (None, 3, None, None, ValueError, "out-of-range"),
        (None, -4, None, None, ValueError, "out-of-range"),
    ],
)
def test_prepare_time_varying_at_steps_rejects_invalid_inputs(
    mutate, time_steps, variable_keys, variable_names, error, match
):
    targets, estimates = time_varying_data()
    if mutate == "bad_target":
        targets["theta"] = np.ones((2, 3))
    elif mutate == "bad_estimate":
        estimates["theta"] = np.ones((2, 4, 3))
    elif mutate == "inconsistent_pair":
        estimates["theta"] = np.ones((3, 4, 3, 1))
    elif mutate == "inconsistent_variables":
        targets["eta"] = np.ones((2, 2, 1))
        estimates["eta"] = np.ones((2, 4, 2, 1))

    workflow = bare_workflow(SimpleNamespace(local_keys=["theta", "eta"]))
    with pytest.raises(error, match=match):
        workflow._prepare_time_varying_at_steps(targets, estimates, time_steps, variable_keys, variable_names)


def test_at_steps_plot_wrappers_forward_prepared_arrays(monkeypatch):
    targets, estimates = time_varying_data(keys=("theta",))
    workflow = bare_workflow(SimpleNamespace(local_keys=["theta"]))
    recovery = Mock(return_value="recovery")
    calibration = Mock(return_value="calibration")
    z_score = Mock(return_value="z-score")
    monkeypatch.setattr(workflow_module, "plot_recovery", recovery)
    monkeypatch.setattr(workflow_module, "plot_calibration", calibration)
    monkeypatch.setattr(workflow_module, "plot_z_score_contraction", z_score)

    assert workflow.recovery_at_steps(targets, estimates, 1, alpha=0.2) == "recovery"
    assert workflow.calibration_at_steps(targets, estimates, 1, alpha=0.3) == "calibration"
    assert workflow.z_score_contraction_at_steps(targets, estimates, 1, alpha=0.4) == "z-score"
    assert recovery.call_args.kwargs["variable_names"] == ["theta"]
    assert calibration.call_args.kwargs["targets"].shape == (2, 1)
    assert z_score.call_args.kwargs["estimates"].shape == (2, 4, 1)


def patch_verification_plots(monkeypatch):
    plots = {
        "recovery": Mock(return_value="recovery"),
        "calibration": Mock(return_value="calibration"),
        "z_score": Mock(return_value="z-score"),
    }
    monkeypatch.setattr(workflow_module, "plot_recovery", plots["recovery"])
    monkeypatch.setattr(workflow_module, "plot_calibration", plots["calibration"])
    monkeypatch.setattr(workflow_module, "plot_z_score_contraction", plots["z_score"])
    return plots


def test_verify_time_invariant_forwards_array_inputs(monkeypatch):
    plots = patch_verification_plots(monkeypatch)
    targets = np.ones((2, 1))
    estimates = np.ones((2, 4, 1))

    result = bare_workflow().verify_time_invariant(
        targets, estimates, variable_names=["Theta"], uncertainty_agg=np.std, color="red"
    )

    assert result == ("recovery", "calibration", "z-score")
    assert plots["recovery"].call_args.kwargs["uncertainty_agg"] is np.std
    assert "uncertainty_agg" not in plots["calibration"].call_args.kwargs
    assert plots["z_score"].call_args.kwargs["color"] == "red"


def test_verify_time_invariant_expands_mixture_components(monkeypatch):
    plots = patch_verification_plots(monkeypatch)
    mixture = SimpleNamespace(names=["fast", "slow"])
    model = SimpleNamespace(
        hyper_keys=["theta", "mix_mixture_weights"],
        shared_keys=[],
        prior=SimpleNamespace(params={"mix": mixture}),
    )
    workflow = bare_workflow(model)
    estimates = {
        "theta": np.ones((2, 3, 4, 1)),
        "mix_mixture_weights": np.ones((2, 3, 4, 2)),
    }
    targets = {"theta": np.array([1.0, 2.0]), "mix_mixture_weights": np.ones((2, 2))}

    result = workflow.verify_time_invariant(targets, estimates)

    assert result == ("recovery", "calibration", "z-score")
    call = plots["recovery"].call_args.kwargs
    assert call["targets"].shape == (2, 3)
    assert call["estimates"].shape == (2, 12, 3)
    assert call["variable_names"] == ["theta", "mix_mixture_weights_fast", "mix_mixture_weights_slow"]


def test_verify_time_invariant_validates_mapping_selection(monkeypatch):
    patch_verification_plots(monkeypatch)
    workflow = bare_workflow(SimpleNamespace(hyper_keys=[], shared_keys=[], prior=SimpleNamespace(params={})))
    with pytest.raises(ValueError, match="No time-invariant"):
        workflow.verify_time_invariant({}, {})

    with pytest.raises(ValueError, match="not found"):
        workflow.verify_time_invariant({}, {}, variable_keys=["theta"])

    estimates = {"theta": np.ones((2, 3, 4, 1))}
    targets = {"theta": np.ones(2)}
    with pytest.raises(ValueError, match="variable_names"):
        workflow.verify_time_invariant(targets, estimates, variable_keys=["theta"], variable_names=[])


@pytest.mark.parametrize(
    ("values", "batch_size", "components", "expected", "match"),
    [
        (np.ones(3), 2, 1, None, "batch size"),
        (np.ones(2), 2, 2, None, "must have 2 components"),
        (np.ones((2, 3, 2)), 2, 1, None, "must have shape"),
        (np.array([[1.0, 2.0], [1.0, 3.0]]), 2, 1, None, "varies across steps"),
        (np.array([[1.0, 1.0], [2.0, 2.0]]), 2, 1, np.array([[1.0], [2.0]]), None),
        (np.ones((2, 3, 2)), 2, 2, np.ones((2, 2)), None),
    ],
)
def test_normalize_time_invariant_target(values, batch_size, components, expected, match):
    if match:
        with pytest.raises(ValueError, match=match):
            Workflow._normalize_time_invariant_target("theta", values, batch_size, components)
    else:
        result = Workflow._normalize_time_invariant_target("theta", values, batch_size, components)
        np.testing.assert_array_equal(result, expected)


def test_posterior_plot_wrappers_supply_model_defaults(monkeypatch):
    varying_plot = Mock(return_value="varying")
    invariant_plot = Mock(return_value="invariant")
    monkeypatch.setattr(workflow_module, "plot_time_varying_posterior", varying_plot)
    monkeypatch.setattr(workflow_module, "plot_time_invariant_posterior", invariant_plot)
    mixture_names = {"mix": ["a", "b"]}
    model = SimpleNamespace(
        local_keys=["theta"],
        hyper_keys=["sigma"],
        shared_keys=["tau"],
        prior=SimpleNamespace(_mixture_names=Mock(return_value=mixture_names)),
    )
    workflow = bare_workflow(model)

    assert workflow.plot_time_varying_posterior(np.ones((1, 2, 3, 1)), color="red") == "varying"
    assert workflow.plot_time_invariant_posterior(np.ones((1, 2, 3, 2)), color="blue") == "invariant"
    assert varying_plot.call_args.kwargs["variable_keys"] == ["theta"]
    assert varying_plot.call_args.kwargs["color"] == "red"
    assert invariant_plot.call_args.kwargs["variable_keys"] == ["sigma", "tau"]
    assert invariant_plot.call_args.kwargs["mixture_names"] == mixture_names


@pytest.mark.parametrize(
    ("workflow", "frame", "mapping", "time_col", "error", "match"),
    [
        (bare_workflow(), pd.DataFrame({"id": [1], "x": [1]}), {"x": "x"}, None, AttributeError, "with a model"),
        (
            bare_workflow(SimpleNamespace(data_keys=["x"])),
            pd.DataFrame({"id": [1]}),
            {"x": "x"},
            None,
            KeyError,
            "missing required",
        ),
        (
            bare_workflow(SimpleNamespace(data_keys=["x"])),
            pd.DataFrame(columns=["id", "x"]),
            {"x": "x"},
            None,
            ValueError,
            "at least one row",
        ),
        (
            bare_workflow(SimpleNamespace(data_keys=["x", "y"])),
            pd.DataFrame({"id": [1], "x": [1]}),
            {"x": "x"},
            None,
            ValueError,
            "do not match",
        ),
        (
            bare_workflow(SimpleNamespace(data_keys=["x"])),
            pd.DataFrame({"id": [1, 1], "time": [0, 0], "x": [1, 2]}),
            {"x": "x"},
            "time",
            ValueError,
            "Duplicate",
        ),
        (
            bare_workflow(SimpleNamespace(data_keys=["x"])),
            pd.DataFrame({"id": [1], "x": ["invalid"]}),
            {"x": "x"},
            None,
            ValueError,
            "must be numeric",
        ),
    ],
)
def test_prepare_data_rejects_invalid_inputs(workflow, frame, mapping, time_col, error, match):
    with pytest.raises(error, match=match):
        workflow.prepare_data(frame, "id", mapping, time_col=time_col)


def test_prepare_data_supports_per_variable_missing_values():
    model = SimpleNamespace(
        data_keys=["x", "y"],
        missing=SimpleNamespace(missing_value={"x": -10.0, "y": -20.0}),
    )
    workflow = bare_workflow(model)
    frame = pd.DataFrame({"id": [1, 1], "x": [1.0, np.nan], "y": [2.0, 3.0]})

    data = workflow.prepare_data(frame, "id", {"x": "x", "y": "y"}, missing_value=np.nan)

    np.testing.assert_array_equal(data["missing_mask"], [[False, True]])
    np.testing.assert_array_equal(data["x"], [[1.0, -10.0]])
    np.testing.assert_array_equal(data["y"], [[2.0, -20.0]])


def test_prepare_data_supports_array_missing_values_and_rejects_bad_shape():
    frame = pd.DataFrame({"id": [1], "x": [-1.0], "y": [2.0]})
    model = SimpleNamespace(
        data_keys=["x", "y"],
        missing=SimpleNamespace(missing_value=np.array([-10.0, -20.0])),
    )
    workflow = bare_workflow(model)
    data = workflow.prepare_data(frame, "id", {"x": "x", "y": "y"}, missing_value=-1.0)
    np.testing.assert_array_equal(data["x"], [[-10.0]])
    np.testing.assert_array_equal(data["y"], [[-20.0]])

    workflow.model.missing.missing_value = np.ones(3)
    with pytest.raises(ValueError, match="scalar, mapping, or shape"):
        workflow.prepare_data(frame, "id", {"x": "x", "y": "y"}, missing_value=-1.0)
