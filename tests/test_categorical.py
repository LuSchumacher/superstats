"""Categorical terms use fixed schemas and preserve named coefficient semantics."""

import copy
import json
import pickle
from dataclasses import FrozenInstanceError

import bayesflow as bf
import keras
import numpy as np
import pandas as pd
import pytest

from superstats import Categorical, Formula, JointPrior, LinkFunction, Model, Prior, Workflow
from superstats.transition import Linear, RandomWalk


def formula():
    return Formula(
        ["v = v_0 + C(condition, low=b_low, high=b_high)"],
        categorical={"condition": Categorical(["control", "low", "high"], reference="control")},
    )


def model(*, missing=None, **kwargs):
    return Model(
        JointPrior(v_0=Prior("normal", loc=1, scale=0), b_low=2, b_high=4),
        lambda v: {"observation": v},
        formula=formula(),
        context_simulator={"condition": ["control", "low", "high"]},
        design_context=("condition",),
        missing=missing,
        **kwargs,
    )


@pytest.mark.parametrize(
    "levels,reference,error",
    [
        ("abc", "a", TypeError),
        ({"a", "b"}, "a", TypeError),
        ([], "a", ValueError),
        (["a"], "a", ValueError),
        (["a", "a"], "a", ValueError),
        ([1, 1.0], 1, ValueError),
        (["a", "b"], "c", ValueError),
        ([0, 1], False, ValueError),
        (["a", 1], "a", ValueError),
        ([True, 1], True, ValueError),
        ([None, "a"], "a", ValueError),
        ([float("nan"), 1], 1, ValueError),
        ([float("inf"), 1], 1, ValueError),
        ([[1], [2]], [1], ValueError),
        (np.array([[1, 2]]), 1, ValueError),
    ],
)
def test_invalid_schemas(levels, reference, error):
    with pytest.raises(error):
        Categorical(levels, reference)


def test_schema_is_immutable_and_json_serializable():
    levels = [np.int64(0), np.int64(1), np.int64(2)]
    schema = Categorical(levels, reference=np.int64(1))
    levels[0] = 99
    assert schema.levels == (0, 1, 2)
    assert schema.non_reference_levels == (0, 2)
    assert Categorical(**json.loads(json.dumps(schema.get_config()))) == schema
    with pytest.raises(FrozenInstanceError):
        schema.reference = 0


def test_formula_selects_effects_and_reports_real_dependencies():
    resolver = formula()
    resolved = resolver.resolve(
        {"v_0": 1, "b_low": 2, "b_high": 4},
        {"condition": ["control", "high", "low", "control"]},
    )
    np.testing.assert_array_equal(resolved["v"], [1, 5, 3, 1])
    assert resolver.dependencies == {"v": {"v_0", "condition", "b_low", "b_high"}}
    assert set(resolved) == {"v_0", "b_low", "b_high", "v"}
    resolver.categorical.clear()
    assert "condition" in resolver.categorical


@pytest.mark.parametrize("clone", [copy.deepcopy, lambda value: pickle.loads(pickle.dumps(value))])
def test_formula_survives_copy_and_pickle(clone):
    resolved = clone(formula()).resolve({"v_0": 1, "b_low": 2, "b_high": 4}, {"condition": "low"})
    assert resolved["v"] == 3


@pytest.mark.parametrize("reference", ["control", "low", "high"])
def test_reference_is_explicit_and_effect_order_does_not_matter(reference):
    levels = ["control", "low", "high"]
    effects = {level: index + 1 for index, level in enumerate(levels) if level != reference}
    resolver = Formula(
        [f"v = 10 + C(condition, {effects!r})"],
        categorical={"condition": Categorical(levels, reference)},
    )
    np.testing.assert_array_equal(
        resolver.resolve({}, {"condition": levels})["v"], [10 + effects.get(level, 0) for level in levels]
    )


@pytest.mark.parametrize(
    "levels,reference,term,values,expected",
    [
        ([-1, 0, 2], 0, "C(x, {-1: b, 2: 2 * b})", [2, -1, 0], [6, 3, 0]),
        (["control", "very low"], "control", "C(x, {'very low': b})", ["very low", "control"], [3, 0]),
        ([False, True], False, "C(x, {True: b})", [False, True], [0, 3]),
        ([0.5, 1.5], 0.5, "C(x, {1.5: b})", [1.5, 0.5], [3, 0]),
    ],
)
def test_literal_mapping_supports_numeric_boolean_and_arbitrary_string_labels(
    levels, reference, term, values, expected
):
    resolver = Formula([f"v = {term}"], categorical={"x": Categorical(levels, reference)})
    np.testing.assert_array_equal(resolver.resolve({"b": 3}, {"x": values})["v"], expected)


def test_shared_and_time_varying_effects_broadcast_and_compose_with_interactions():
    resolver = Formula(
        ["offset = C(condition, high=b_high, low=b_low) * x", "v = v_0 + offset"],
        categorical=formula().categorical,
    )
    resolved = resolver.resolve(
        {"v_0": np.array([1, 10]), "b_low": np.array([2, 3]), "b_high": [[4, 5, 6], [7, 8, 9]]},
        {"condition": [["control", "low", "high"], ["high", "control", "low"]], "x": [[1, 2, 3], [1, 2, 3]]},
    )
    np.testing.assert_array_equal(resolved["v"], [[1, 5, 19], [17, 10, 19]])


def test_two_categorical_predictors_and_numeric_interactions():
    resolver = Formula(
        ["v = 1 + C(a, low=b) * C(c, {2: d})"],
        categorical={"a": Categorical(["control", "low"], "control"), "c": Categorical([0, 2], 0)},
    )
    result = resolver.resolve({"b": 2, "d": 3}, {"a": ["control", "low", "low"], "c": [2, 0, 2]})
    np.testing.assert_array_equal(result["v"], [1, 1, 7])


def test_nested_terms_assign_independent_interaction_coefficients():
    resolver = Formula(
        ["v = C(a, treatment=C(b, low=b_low, high=b_high))"],
        categorical={
            "a": Categorical(["control", "treatment"], "control"),
            "b": Categorical(["control", "low", "high"], "control"),
        },
    )
    result = resolver.resolve(
        {"b_low": 2, "b_high": 3},
        {"a": ["control", "treatment", "treatment", "treatment"], "b": ["high", "control", "low", "high"]},
    )
    np.testing.assert_array_equal(result["v"], [0, 0, 2, 3])


def test_call_name_is_not_confused_with_a_coefficient_named_c():
    resolver = Formula(["v = C(x, low=C)"], categorical={"x": Categorical(["control", "low"], "control")})
    assert resolver.dependencies["v"] == {"x", "C"}
    assert resolver.resolve({"C": 3}, {"x": "low"})["v"] == 3


@pytest.mark.parametrize("mask", [np.array([True]), np.array([[0, 2]])])
def test_categorical_encoding_rejects_invalid_masks(mask):
    with pytest.raises(ValueError, match="mask"):
        Categorical(["a", "b"], "a").encode([["a", "b"]], missing_mask=mask)


def test_numeric_labels_are_nominal_and_do_not_accept_string_or_boolean_aliases():
    schema = Categorical([0, 5, 10], reference=5)
    np.testing.assert_array_equal(schema.encode([10, 0, 5]), [[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    for invalid in [["0"], [False]]:
        with pytest.raises(ValueError, match="category"):
            schema.encode(invalid)


def test_fixed_context_and_adapter_do_not_coerce_numeric_values_to_string_labels():
    resolver = Formula(["v = C(x, low=2)"], categorical={"x": Categorical(["1", "low"], "1")})
    simulator = Model(
        JointPrior(),
        lambda v: {"observation": v},
        formula=resolver,
        design_context=("x",),
        context_simulator={"x": ["1", "low"]},
        missing=None,
    )
    with pytest.raises(ValueError, match="category"):
        simulator.simulate_from_parameters({}, 1, 2, context={"x": [1, "low"]})
    adapter = Workflow.default_adapter(simulator)
    with pytest.raises(ValueError, match="category"):
        adapter({"time_steps": [[1, 2]], "observation": [[0, 2]], "x": [[1, "low"]]})


def test_unused_levels_do_not_change_shape_or_hide_missing_coefficients():
    resolver = formula()
    result = resolver.resolve({"v_0": [1, 2], "b_low": [2, 3], "b_high": [4, 5]}, {"condition": [["control"], ["low"]]})
    np.testing.assert_array_equal(result["v"], [[1], [5]])
    with pytest.raises(KeyError, match="b_high"):
        resolver.resolve({"v_0": 1, "b_low": 2}, {"condition": "control"})


@pytest.mark.parametrize("values", [["unknown"], [None], [np.nan], [pd.NA], [0], [True]])
def test_unknown_or_missing_categories_fail_explicitly(values):
    with pytest.raises(ValueError, match="category.*condition"):
        formula().resolve({"v_0": 1, "b_low": 2, "b_high": 4}, {"condition": values})


def test_categorical_predictors_must_come_from_context():
    with pytest.raises(KeyError, match="supplied in context"):
        formula().resolve({"v_0": 1, "b_low": 2, "b_high": 4, "condition": "low"})


@pytest.mark.parametrize(
    "term",
    [
        "condition",
        "2 * condition",
        "C(condition)",
        "C(condition, low=b_low)",
        "C(condition, low=b_low, high=b_high, control=0)",
        "C(condition, low=b_low, high=b_high, typo=2)",
        "C(condition, low=b_low, low=b_high, high=2)",
        "C(condition, {'low': b_low, 'low': b_high, 'high': 2})",
        "C(condition, {1: b_low, 2: b_high})",
        "C(condition, **effects)",
        "C(condition, {*effects: b_low})",
        "C(condition, {level: b_low})",
        "C(condition, low=__import__('os'), high=0)",
        "C(condition, low=x[0], high=0)",
        "C(condition, low=b_low.real, high=0)",
        "C(condition, low=b_low > 0, high=0)",
        "C(condition, low='b_low', high=0)",
        "C(condition, {'low': b_low, 'high': b_high}, low=2)",
        "C(condition, b_low, b_high)",
        "C('condition', low=b_low, high=b_high)",
        "C(other, low=b_low, high=b_high)",
        "other(condition, low=b_low, high=b_high)",
        "np.C(condition, low=b_low, high=b_high)",
    ],
)
def test_categorical_syntax_is_strict_and_safe(term):
    with pytest.raises(ValueError):
        Formula([f"v = {term}"], categorical=formula().categorical)


@pytest.mark.parametrize(
    "formulas,categorical,error",
    [
        (["v = C(x, low=b)"], None, ValueError),
        (["v = 1"], ["x"], TypeError),
        (["v = 1"], {"x": [0, 1]}, TypeError),
        (["v = 1"], {"bad name": Categorical([0, 1], 0)}, ValueError),
        (["v = 1"], {"x": Categorical([0, 1], 0)}, ValueError),
        (["x = C(x, low=b)"], {"x": Categorical(["control", "low"], "control")}, ValueError),
    ],
)
def test_invalid_categorical_declarations(formulas, categorical, error):
    with pytest.raises(error):
        Formula(formulas, categorical=categorical)


def test_model_resolves_labels_and_links_but_keeps_raw_context_and_inference_targets():
    simulator = model(formula_link_function={"v": LinkFunction("exp")})
    sample = simulator.sample(2, 3)
    np.testing.assert_allclose(sample["observation"], np.tile(np.exp([1, 3, 5]), (2, 1)))
    np.testing.assert_array_equal(sample["condition"], np.tile(["control", "low", "high"], (2, 1)))
    assert simulator.shared_keys == ["v_0"]
    assert simulator.formula_keys == ["v"]
    assert simulator.summary_keys == ["observation", "condition"]


@pytest.mark.parametrize(
    "transition",
    [Linear(intercept=2, slope=2), RandomWalk(initial_prior=Prior("normal", loc=2, scale=0), sigma=0, delta=1)],
)
def test_model_time_varying_categorical_effects(transition):
    simulator = Model(
        JointPrior(v_0=1, b_low=transition, b_high=4),
        lambda v: {"observation": v},
        formula=formula(),
        context_simulator=lambda batch_size, num_steps: {"condition": np.full((batch_size, num_steps), "low")},
        design_context=("condition",),
        latent_link_function={"b_low": LinkFunction("exp")},
        missing=None,
    )
    sample = simulator.sample(2, 3)
    expected = np.tile(1 + np.exp([2, 3, 4]), (2, 1))
    np.testing.assert_allclose(sample["observation"], expected)


def test_categorical_predictors_must_be_selected_as_design_context():
    with pytest.raises(ValueError, match="design_context"):
        Model(JointPrior(v_0=1, b_low=2, b_high=4), lambda v: {"x": v}, formula=formula(), missing=None)


def test_adapter_one_hot_order_is_fixed_and_checkpoint_safe():
    simulator = model()
    sample = simulator.sample(2, 3)
    adapter = Workflow.default_adapter(simulator)
    adapted = adapter(sample)
    assert adapted["summary_variables"].shape == (2, 3, 5)  # time, observation, three levels
    np.testing.assert_array_equal(adapted["summary_variables"][..., 2:], np.tile(np.eye(3), (2, 1, 1)))
    # Warm the adapter so inference-target concatenations can also be inverted.
    restored = keras.saving.deserialize_keras_object(
        json.loads(json.dumps(keras.saving.serialize_keras_object(adapter)))
    )
    np.testing.assert_array_equal(restored(sample)["summary_variables"], adapted["summary_variables"])
    inverse = restored.inverse({"invariant_variables": adapted["invariant_variables"]})
    np.testing.assert_array_equal(inverse["v_0"], sample["v_0"])
    labels = restored.inverse(adapted)["condition"]
    np.testing.assert_array_equal(labels, sample["condition"])

    sample["condition"] = np.full((2, 3), "high")
    absent_levels = restored(sample)["summary_variables"][..., 2:]
    np.testing.assert_array_equal(absent_levels, np.tile([0, 0, 1], (2, 3, 1)))


@pytest.mark.parametrize("missing_value", [-1, np.nan, -99999])
def test_missingness_preserves_string_labels_and_encodes_masked_rows_as_zero(missing_value):
    from superstats.simulation import RandomMissingProcess

    simulator = model(missing=RandomMissingProcess(missing_value=missing_value))
    simulator.prior.params["p_missing"] = 0.5
    sample = simulator.sample(8, 3, rng=np.random.default_rng(10))
    mask = sample["missing_mask"]
    assert mask.any() and (~mask).any()
    if np.isnan(missing_value):
        assert pd.isna(sample["condition"][mask]).all()
    else:
        assert (sample["condition"][mask] == missing_value).all()
    adapted = Workflow.default_adapter(simulator)(sample)
    encoded = adapted["summary_variables"][..., 2:5]
    assert (encoded[mask] == 0).all()
    assert (encoded[~mask].sum(axis=-1) == 1).all()
    np.testing.assert_array_equal(adapted["summary_variables"][..., -1], mask)


def test_prepare_data_accepts_raw_labels_and_masks_padding_and_missing_observations():
    from superstats.simulation import RandomMissingProcess

    simulator = model(missing=RandomMissingProcess(missing_value=-999))
    workflow = Workflow.__new__(Workflow)
    workflow.model = simulator
    frame = pd.DataFrame({"id": [1, 1, 2], "y": [1, -1, 5], "group": ["control", pd.NA, "high"]})
    data = workflow.prepare_data(frame, "id", {"y": "observation", "group": "condition"}, missing_value=-1)
    np.testing.assert_array_equal(data["condition"], np.array([["control", -999], ["high", -999]], dtype=object))
    np.testing.assert_array_equal(data["missing_mask"], [[False, True], [False, True]])
    adapter = Workflow.default_adapter(simulator)
    adapter(simulator.sample(2, 3))
    adapted = adapter(workflow._prepare_conditions(data), strict=False)
    np.testing.assert_array_equal(
        adapted["summary_variables"][..., 2:5], [[[1, 0, 0], [0, 0, 0]], [[0, 0, 1], [0, 0, 0]]]
    )

    for unknown in ["unknown", pd.NA, np.nan]:
        invalid = frame.copy()
        invalid.loc[0, "group"] = unknown
        with pytest.raises(ValueError, match="category.*condition"):
            workflow.prepare_data(invalid, "id", {"y": "observation", "group": "condition"}, missing_value=-1)


def test_resimulation_uses_original_categorical_context_and_dataset_selection():
    workflow = Workflow.__new__(Workflow)
    workflow.model = model()
    context = {"condition": np.array([["control", "low", "high"], ["high", "high", "control"]])}
    estimates = {"v_0": np.array([[[1.0], [1.0]], [[10.0], [10.0]]])}
    result = workflow.resimulate(estimates, context=context, data_idx=[1, 0], num_sims=2, rng=0)
    np.testing.assert_array_equal(result["observation"], [[[14, 14, 10], [14, 14, 10]], [[1, 3, 5], [1, 3, 5]]])
    invalid = {"condition": np.array([["control", "low", "typo"], ["high", "high", "control"]])}
    with pytest.raises(ValueError, match="category.*condition"):
        workflow.resimulate(estimates, context=invalid, num_sims=1)


def test_real_workflow_trains_and_samples_with_categorical_context():
    simulator = Model(
        JointPrior(v_0=Prior("normal"), b_low=Prior("normal"), b_high=Prior("normal")),
        lambda v: {"observation": v + np.random.normal(size=v.shape)},
        formula=formula(),
        context_simulator={"condition": ["control", "low", "high"]},
        design_context=("condition",),
        missing=None,
    )
    workflow = Workflow(
        model=simulator,
        embedding_network=keras.layers.Dense(4),
        invariant_inference_network=bf.networks.CouplingFlow(depth=1, subnet_kwargs={"widths": (8,)}),
    )
    data = simulator.sample(8, 3)
    history = workflow.fit_offline(data, data, epochs=1, batch_size=4, verbose=0)
    assert np.isfinite(history.history["loss"]).all()
    online_history = workflow.fit_online(num_steps=3, epochs=1, num_batches_per_epoch=1, batch_size=4, verbose=0)
    assert np.isfinite(online_history.history["loss"]).all()
    samples = workflow.sample(data, num_samples=2, batch_size=4, seed=7)
    assert set(samples) == {"v_0", "b_low", "b_high"}
    assert all(value.shape == (8, 2, 1) for value in samples.values())
    assert np.isfinite(workflow.approximator.log_prob(data)).all()
