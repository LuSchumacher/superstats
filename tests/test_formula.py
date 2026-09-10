import numpy as np
import pandas as pd
import pytest

from superstats import Formula, JointPrior, Model
from superstats.simulation import ContextMapping, ContextSimulator


def test_formula_resolves_expressions_and_preserves_parameter_draws():
    formula = Formula(
        [
            "v = v_0 + b_v * covariate",
            "tau = tau_0 + b_tau * n_cues",
        ]
    )
    parameters = {
        "v_0": np.array([1.0, 2.0]),
        "b_v": np.array([0.5, 1.0]),
        "tau_0": np.array([0.2, 0.3]),
        "b_tau": np.array([0.1, 0.2]),
        "a": np.array([1.5, 1.5]),
    }
    context = {"covariate": np.array([2.0, 3.0]), "n_cues": np.array([1.0, 2.0])}

    resolved = formula.resolve(parameters=parameters, context=context)

    assert set(resolved) == {*parameters, "v", "tau"}
    np.testing.assert_allclose(resolved["v"], [2.0, 5.0])
    np.testing.assert_allclose(resolved["tau"], [0.3, 0.7])


def test_formula_broadcasts_batch_level_coefficients_over_context_time():
    formula = Formula(["v = v_0 + b_v * covariate"])
    resolved = formula.resolve(
        parameters={"v_0": np.array([1.0, 2.0]), "b_v": np.array([0.5, 1.0])},
        context={"covariate": np.array([[2.0, 4.0, 6.0], [1.0, 3.0, 5.0]])},
    )

    assert resolved["v"].shape == (2, 3)
    np.testing.assert_allclose(resolved["v"], [[2.0, 3.0, 4.0], [3.0, 5.0, 7.0]])


def test_formula_supports_ordered_dependencies():
    formula = Formula(["offset = beta * covariate", "v = v_0 + offset"])

    resolved = formula.resolve({"v_0": 1.0, "beta": 2.0}, {"covariate": np.array([0.0, 1.0])})

    np.testing.assert_allclose(resolved["v"], [1.0, 3.0])


@pytest.mark.parametrize("formula", ["v + b * x", "v = np.exp(x)", "v = x[0]", "v = x > 0"])
def test_formula_rejects_unsafe_or_invalid_expressions(formula):
    with pytest.raises(ValueError):
        Formula([formula])


def test_formula_reports_unknown_names_and_ambiguous_context_names():
    formula = Formula(["v = v_0 + covariate"])
    with pytest.raises(KeyError, match="covariate"):
        formula.resolve({"v_0": 1.0})
    with pytest.raises(ValueError, match="shadow"):
        formula.resolve({"v_0": 1.0}, {"v_0": 2.0, "covariate": 3.0})


def test_model_applies_formula_to_context_simulated_covariates():
    def context_simulator(*, batch_size, num_steps):
        return {"covariate": np.broadcast_to(np.arange(num_steps), (batch_size, num_steps))}

    def simulator(v):
        return {"observation": v}

    model = Model(
        prior=JointPrior(v_0=1.0, b_v=2.0),
        simulator=simulator,
        missing=None,
        context=ContextSimulator(context_simulator),
        context_mapping=ContextMapping(formula_context=("covariate",)),
        formula=Formula(["v = v_0 + b_v * covariate"]),
    )

    sample = model.sample(batch_size=2, num_steps=3)

    np.testing.assert_allclose(sample["observation"], [[1.0, 3.0, 5.0], [1.0, 3.0, 5.0]])
    np.testing.assert_array_equal(sample["covariate"], [[0, 1, 2], [0, 1, 2]])


@pytest.mark.parametrize(
    "context",
    [
        {"covariate": np.array([0.0, 1.0, 2.0])},
        pd.DataFrame({"covariate": [0.0, 1.0, 2.0]}),
    ],
    ids=["mapping", "dataframe"],
)
def test_model_accepts_fixed_context(context):
    def simulator(v):
        return {"observation": v}

    model = Model(
        prior=JointPrior(v_0=1.0),
        simulator=simulator,
        missing=None,
        context=context,
        context_mapping=ContextMapping(formula_context=("covariate",)),
        formula=Formula(["v = v_0 + covariate"]),
    )

    sample = model.sample(batch_size=2, num_steps=3)

    np.testing.assert_allclose(sample["observation"], [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    np.testing.assert_allclose(sample["covariate"], [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])


def test_model_rejects_fixed_context_with_wrong_number_of_trials():
    def simulator(v):
        return {"observation": v}

    model = Model(
        prior=JointPrior(v=1.0),
        simulator=simulator,
        missing=None,
        context={"covariate": [0.0, 1.0]},
    )

    with pytest.raises(ValueError, match="3 trial rows"):
        model.sample(batch_size=1, num_steps=3)
