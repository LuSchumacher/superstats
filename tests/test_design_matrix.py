import numpy as np
import pytest

from superstats import DesignMatrix, JointPrior, Model
from superstats.simulation import ContextMapping, ContextSimulator


def test_design_matrix_resolves_formulas_and_preserves_parameter_draws():
    design = DesignMatrix(
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

    resolved = design.resolve(parameters=parameters, context=context)

    assert set(resolved) == {*parameters, "v", "tau"}
    np.testing.assert_allclose(resolved["v"], [2.0, 5.0])
    np.testing.assert_allclose(resolved["tau"], [0.3, 0.7])


def test_design_matrix_broadcasts_batch_level_coefficients_over_context_time():
    design = DesignMatrix(["v = v_0 + b_v * covariate"])
    resolved = design.resolve(
        parameters={"v_0": np.array([1.0, 2.0]), "b_v": np.array([0.5, 1.0])},
        context={"covariate": np.array([[2.0, 4.0, 6.0], [1.0, 3.0, 5.0]])},
    )

    assert resolved["v"].shape == (2, 3)
    np.testing.assert_allclose(resolved["v"], [[2.0, 3.0, 4.0], [3.0, 5.0, 7.0]])


def test_design_matrix_supports_ordered_formula_dependencies():
    design = DesignMatrix(["offset = beta * covariate", "v = v_0 + offset"])

    resolved = design.resolve({"v_0": 1.0, "beta": 2.0}, {"covariate": np.array([0.0, 1.0])})

    np.testing.assert_allclose(resolved["v"], [1.0, 3.0])


@pytest.mark.parametrize("formula", ["v + b * x", "v = np.exp(x)", "v = x[0]", "v = x > 0"])
def test_design_matrix_rejects_unsafe_or_invalid_formulas(formula):
    with pytest.raises(ValueError):
        DesignMatrix([formula])


def test_design_matrix_reports_unknown_names_and_ambiguous_context_names():
    design = DesignMatrix(["v = v_0 + covariate"])
    with pytest.raises(KeyError, match="covariate"):
        design.resolve({"v_0": 1.0})
    with pytest.raises(ValueError, match="shadow"):
        design.resolve({"v_0": 1.0}, {"v_0": 2.0, "covariate": 3.0})


def test_model_applies_design_matrix_to_context_simulated_covariates():
    def context_simulator(*, batch_size, num_steps):
        return {"covariate": np.broadcast_to(np.arange(num_steps), (batch_size, num_steps))}

    def simulator(v):
        return {"observation": v}

    model = Model(
        prior=JointPrior(v_0=1.0, b_v=2.0),
        simulator=simulator,
        missing=None,
        context=ContextSimulator(context_simulator),
        context_mapping=ContextMapping(design_context=("covariate",)),
        design_matrix=DesignMatrix(["v = v_0 + b_v * covariate"]),
    )

    sample = model.sample(batch_size=2, num_steps=3)

    np.testing.assert_allclose(sample["observation"], [[1.0, 3.0, 5.0], [1.0, 3.0, 5.0]])
    np.testing.assert_array_equal(sample["covariate"], [[0, 1, 2], [0, 1, 2]])
