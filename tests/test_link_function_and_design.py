import matplotlib.pyplot as plt
import numpy as np
import pytest

from superstats import Formula, JointPrior, LinkFunction, Model, Prior
from superstats.transition import Linear, Mixture, RandomWalk, Jump


def simulator(a, v=0.0):
    return {"observation": a + v}


@pytest.mark.parametrize("function,expected", [("softplus", np.log(2)), ("identity", 0), ("exp", 1)])
def test_link_options(function, expected):
    assert LinkFunction(function)(0) == pytest.approx(expected)


def test_scaled_sigmoid_bounds_and_stability():
    assert LinkFunction()(0) == pytest.approx(0.5)
    link = LinkFunction(bounds=(0.2, 4))
    np.testing.assert_allclose(link(np.array([-10000.0, 0.0, 10000.0])), [0.2, 2.1, 4])
    for bounds in [(1, 1), (2, 1), (0, np.inf), (0,)]:
        with pytest.raises(ValueError):
            LinkFunction(bounds=bounds)
    with pytest.raises(ValueError, match="shape"):
        LinkFunction(lambda x: np.zeros(2))(np.zeros(3))
    with pytest.raises(ValueError, match="Unknown link function"):
        LinkFunction("sigmoid")


def test_clip_requires_bounds_and_clips_without_changing_shape():
    with pytest.raises(ValueError, match="bounds are required"):
        LinkFunction("clip")

    link = LinkFunction("clip", bounds=(-1.0, 2.0))
    values = np.array([[-2.0, -1.0, 0.5, 2.0, 3.0]])

    np.testing.assert_array_equal(link(values), [[-1.0, -1.0, 0.5, 2.0, 2.0]])


@pytest.mark.parametrize("bounds", [(1, 1), (2, 1), (0, np.inf), (0,)])
def test_clip_rejects_invalid_bounds(bounds):
    with pytest.raises(ValueError, match="bounds"):
        LinkFunction("clip", bounds=bounds)


def test_model_clips_completed_deterministic_trajectory_but_keeps_raw_target():
    model = Model(
        JointPrior(a=Linear(intercept=-1, slope=4)),
        simulator,
        missing=None,
        link_function={"a": LinkFunction("clip", bounds=(0, 2))},
    )

    sample = model.sample(batch_size=1, num_steps=3)

    np.testing.assert_array_equal(sample["a"][..., 0], [[-1, 1, 3]])
    np.testing.assert_array_equal(sample["observation"], [[0, 1, 2]])


@pytest.mark.parametrize("vary_intercept,vary_slope", [(False, False), (True, False), (False, True), (True, True)])
def test_regression_links_after_mixed_coefficients_and_resimulation(vary_intercept, vary_slope):
    intercept = Linear(intercept=-2, slope=1) if vary_intercept else Prior("normal", loc=-2, scale=0)
    slope = Linear(intercept=1, slope=2) if vary_slope else Prior("normal", loc=1, scale=0)
    resolver = {"formula": Formula(["a = a_0 + b_a * x"])}
    model = Model(
        JointPrior(a_0=intercept, b_a=slope),
        simulator,
        missing=None,
        context={"x": [-2, 0, 2]},
        design_context=("x",),
        link_function={"a": LinkFunction(bounds=(0.2, 4))},
        **resolver,
    )
    draws = model.sample_prior(2, 3)
    raw = {**draws["shared_params"], **draws["deterministic_params"]}
    a0 = raw["a_0"] if vary_intercept else raw["a_0"][:, None]
    ba = raw["b_a"] if vary_slope else raw["b_a"][:, None]
    expected = LinkFunction(bounds=(0.2, 4))(a0 + ba * np.array([-2, 0, 2]))
    np.testing.assert_allclose(draws["model_params"]["a"], expected)
    sample = model.sample(2, 3)
    np.testing.assert_allclose(sample["observation"], expected)
    np.testing.assert_allclose(sample["a"][..., 0], expected)
    assert model.formula_keys == ["a"]
    np.testing.assert_allclose(model.simulate_from_parameters(raw, 2, 3)["observation"], expected)


def test_context_binding_precedes_link_and_defaults_are_linked():
    model = Model(
        JointPrior(),
        simulator,
        missing=None,
        context={"a": [-2, 2]},
        simulator_context=("a",),
        link_function={"a": LinkFunction("softplus"), "v": LinkFunction("exp")},
    )
    expected = np.logaddexp(0, [-2, 2]) + 1
    np.testing.assert_allclose(model.sample(2, 2)["observation"], np.tile(expected, (2, 1)))


def test_formula_interactions_and_link_validation():
    formula = Formula(["a = base + slope * x * y"])
    np.testing.assert_allclose(
        formula.resolve(parameters={"base": 2, "slope": -1}, context={"x": [1, 2], "y": [3, 4]})["a"], [-1, -6]
    )
    with pytest.raises(KeyError, match="unknown name"):
        formula.resolve(parameters={})
    with pytest.raises(ValueError, match="Unknown"):
        Model(JointPrior(a=1), simulator, link_function={"typo": LinkFunction()})


def test_stochastic_and_mixture_outputs_are_raw():
    walk = RandomWalk(initial_prior=Prior("normal", loc=-2, scale=0), sigma=0, delta=1)
    np.testing.assert_allclose(walk.sample(1, 3)["local_params"], [[-2, -1, 0]])
    mixture = Mixture(
        [RandomWalk(sigma=0, delta=1), Jump()], mixture_weights=(1, 0), initial_prior=Prior("normal", loc=-2, scale=0)
    )
    np.testing.assert_allclose(mixture.sample(1, 3)["local_params"], [[-2, -1, 0]])


def test_prior_plots_keep_raw_inference_scale():
    model = Model(
        JointPrior(a=Prior("normal", loc=-2, scale=0)),
        simulator,
        missing=None,
        link_function={"a": LinkFunction(bounds=(0.2, 4))},
    )
    raw = model._sample_inference_prior(2, 3)
    np.testing.assert_allclose(raw["shared_params"]["a"], -2)
    fig = model.plot_time_invariant_prior(num_draws=3)
    plt.close(fig)


def test_workflow_resimulation_uses_original_context_and_fixed_coefficients():
    from superstats import Workflow

    model = Model(
        JointPrior(a_0=Prior("normal"), b_a=2),
        simulator,
        missing=None,
        context={"x": [0, 0, 0]},
        design_context=("x",),
        formula=Formula(["a = a_0 + b_a * x"]),
        link_function={"a": LinkFunction("softplus")},
    )
    workflow = Workflow.__new__(Workflow)
    workflow.model = model
    estimates = {"a_0": np.full((2, 4, 1), -2.0)}
    context = {"x": np.array([[-1.0, 0, 1], [1.0, 2, 3]])}
    result = workflow.resimulate(estimates, num_sims=2, rng=0, data_idx=[1, 0], context=context)
    expected = np.logaddexp(0, -2 + 2 * context["x"][[1, 0]])
    np.testing.assert_allclose(result["observation"], np.repeat(expected[:, None, :], 2, axis=1))


def test_model_transforms_contamination_and_retains_raw_targets():
    from superstats.simulation import RandomChoiceContamination

    def diffusion(a):
        return {"response_time": np.ones_like(a), "choice": np.zeros_like(a)}

    process = RandomChoiceContamination(infer=True)
    model = Model(
        JointPrior(p_contaminated=RandomWalk(initial_prior=Prior("normal", loc=-2, scale=0), sigma=0, delta=0), a=1),
        diffusion,
        missing=None,
        contamination=process,
        link_function={"p_contaminated": LinkFunction()},
    )
    sample = model.sample(2, 3)
    np.testing.assert_allclose(sample["p_contaminated"], -2)


def test_shared_only_resimulation_infers_trial_count_from_configured_dataframe():
    import pandas as pd
    from superstats import Workflow

    model = Model(
        JointPrior(a_0=Prior("normal"), b_a=1),
        simulator,
        missing=None,
        context=pd.DataFrame({"x": [-1, 0, 1]}),
        design_context=("x",),
        formula=Formula(["a = a_0 + b_a * x"]),
        link_function={"a": LinkFunction("softplus")},
    )
    workflow = Workflow.__new__(Workflow)
    workflow.model = model
    estimates = {"a_0": np.zeros((1, 2, 1))}
    result = workflow.resimulate(estimates, num_sims=1, rng=0)
    np.testing.assert_allclose(result["observation"], np.logaddexp(0, [[[-1, 0, 1]]]))


def test_deterministic_contamination_reconstructs_during_resimulation():
    from superstats import Workflow
    from superstats.simulation import RandomChoiceContamination

    process = RandomChoiceContamination(infer=True)
    model = Model(
        JointPrior(p_contaminated=Linear(intercept=Prior("normal"), slope=0), a=1),
        lambda a: {"response_time": np.ones_like(a), "choice": np.zeros_like(a)},
        missing=None,
        contamination=process,
        link_function={"p_contaminated": LinkFunction()},
    )
    workflow = Workflow.__new__(Workflow)
    workflow.model = model
    estimates = {"p_contaminated_intercept": np.zeros((1, 2, 1))}
    result = workflow.resimulate(estimates, num_sims=1, num_steps=3, rng=0)
    np.testing.assert_allclose(result["response_time"], np.ones((1, 1, 3)))


def test_joint_plot_retains_transition_hyperparameters_for_derived_targets():
    prior = JointPrior(a_0=RandomWalk(sigma=Prior("halfnormal", scale=0.1)), b_a=Prior("normal"))
    resolver = {"formula": Formula(["intermediate = a_0 + b_a * x", "a = intermediate"])}
    model = Model(
        prior,
        simulator,
        missing=None,
        context={"x": [0, 1]},
        design_context=("x",),
        link_function={"a": LinkFunction()},
        **resolver,
    )
    assert model._sample_inference_prior(2, 2)["hyper_param_groups"]["a_0"] == ["a_0_sigma"]
    fig = model.plot_joint_prior(num_steps=2, num_draws=10, num_trajectories=2)
    assert len(fig.axes) >= 2
    plt.close(fig)


def test_model_routes_callable_context_with_requested_api():
    calls = []
    received = []

    def context_generator(*, batch_size, num_steps):
        calls.append((batch_size, num_steps))
        shape = (batch_size, num_steps)
        return {
            "difficulty": np.full(shape, -2.0),
            "stimulus": np.full(shape, 7),
            "correct_idx": np.ones(shape, dtype=int),
            "unused": np.full(shape, 9),
        }

    def observation_simulator(a, correct_idx, *, context):
        received.append(context)
        return {"observation": a + correct_idx + context["stimulus"].reshape(-1)}

    model = Model(
        prior=JointPrior(a_0=Prior("normal", loc=-2, scale=0), b_a=Linear(intercept=1, slope=0)),
        simulator=observation_simulator,
        context=context_generator,
        formula=Formula(["a = a_0 + b_a * difficulty"]),
        design_context=("difficulty",),
        simulator_context=("stimulus", "correct_idx", "difficulty"),
        link_function={"a": LinkFunction(bounds=(0.2, 4.0))},
        missing=None,
    )
    result = model.sample(2, 3)
    expected = LinkFunction(bounds=(0.2, 4.0))(-4) + 8
    np.testing.assert_allclose(result["observation"], expected)
    assert calls == [(1, 1), (2, 3)]
    assert set(received[-1]) == {"stimulus", "difficulty"}
    np.testing.assert_allclose(received[-1]["difficulty"], -2)
    assert "unused" in result
    explicit = model.simulate_from_parameters(
        {"a_0": np.full(2, -2), "b_a": np.ones((2, 3))},
        2,
        3,
        context=context_generator(batch_size=2, num_steps=3),
    )
    np.testing.assert_allclose(explicit["observation"], expected)


@pytest.mark.parametrize("selector", ["design_context", "simulator_context"])
def test_model_validates_context_selectors(selector):
    with pytest.raises(TypeError, match=selector):
        Model(JointPrior(a=1), simulator, context={"x": [0]}, **{selector: "x"})
    with pytest.raises(KeyError, match="missing"):
        Model(JointPrior(a=1), simulator, context={"x": [0]}, **{selector: ("missing",)})
    with pytest.raises(ValueError, match="require context"):
        Model(JointPrior(a=1), simulator, **{selector: ("x",)})


def test_prior_plots_show_regression_targets_without_context_or_links(monkeypatch):
    import superstats.simulation.model as model_module

    model = Model(
        JointPrior(v_diff_0=RandomWalk(sigma=Prior("halfnormal", scale=0.1)), b_difficulty=Prior("normal"), fixed=1),
        lambda v_diff, correct_idx: {"observation": v_diff},
        missing=None,
        context={"difficulty": [-1, 1], "correct_idx": [0, 1]},
        design_context=("difficulty",),
        simulator_context=("correct_idx",),
        formula=Formula(["v_diff = v_diff_0 + b_difficulty * difficulty"]),
        link_function={"v_diff": LinkFunction(bounds=(0, 3))},
    )
    captured = {}
    monkeypatch.setattr(model_module, "plot_joint_prior", lambda **kwargs: captured.update(kwargs))

    def forbidden(*args, **kwargs):
        raise AssertionError("Inference prior plots must not resolve the simulation pipeline.")

    model._resolve_parameters = forbidden
    model._generate_context = forbidden
    model.plot_joint_prior(num_steps=5, num_draws=3, num_trajectories=2)
    assert set(captured["local_params"]) == {"v_diff_0"}
    assert set(captured["shared_params"]) == {"b_difficulty"}
    assert set(captured["hyper_params"]) == {"v_diff_0_sigma"}
    assert captured["param_bounds"] == {}


@pytest.mark.parametrize("infer", [False, True])
def test_prior_plots_include_only_inferred_contamination(infer):
    from superstats.simulation import RandomChoiceContamination

    model = Model(
        JointPrior(p_contaminated=Prior("beta", a=2, b=8), a=Prior("normal")),
        simulator,
        missing=None,
        contamination=RandomChoiceContamination(infer=infer),
    )
    draws = model._sample_inference_prior(3, 5)
    assert ("p_contaminated" in draws["shared_params"]) == infer


def test_joint_prior_plot_shows_deterministic_trajectories_and_inferred_curve_coefficients(monkeypatch):
    import superstats.simulation.model as model_module

    model = Model(JointPrior(a=Linear(intercept=Prior("normal"), slope=1)), simulator, missing=None)
    captured = {}
    monkeypatch.setattr(model_module, "plot_joint_prior", lambda **kwargs: captured.update(kwargs))
    model.plot_joint_prior(num_steps=4, num_draws=3, num_trajectories=2)
    assert set(captured["local_params"]) == {"a"}
    assert captured["local_params"]["a"].shape == (2, 4)
    assert captured["shared_params"] == {}
    assert set(captured["hyper_params"]) == {"a_intercept"}
    fig = model_module.plot_time_invariant_prior(captured["hyper_params"], {})
    plt.close(fig)


@pytest.mark.parametrize("infer", [False, True])
def test_contamination_probability_sampled_once_and_linked_once(monkeypatch, infer):
    from superstats.simulation import RandomChoiceContamination

    probability = Prior("normal", loc=-2, scale=0)
    process = RandomChoiceContamination(infer=infer)
    model = Model(
        JointPrior(a=1, p_contaminated=probability),
        lambda a: {"response_time": np.ones_like(a), "choice": np.zeros_like(a)},
        link_function={"p_contaminated": LinkFunction()},
        contamination=process,
        missing=None,
    )
    calls = []
    original_sample = probability.sample
    original_apply = process.apply

    def sample(*args, **kwargs):
        calls.append("sample")
        return original_sample(*args, **kwargs)

    def apply(data, **kwargs):
        np.testing.assert_allclose(kwargs["probability"], 1 / (1 + np.exp(2)))
        calls.append("apply")
        return original_apply(data, **kwargs)

    monkeypatch.setattr(probability, "sample", sample)
    monkeypatch.setattr(process, "apply", apply)
    model.sample(2, 3)
    assert calls == ["sample", "apply"]


@pytest.mark.parametrize("infer", [False, True])
def test_resimulation_uses_posterior_or_fresh_nuisance_probability(monkeypatch, infer):
    from superstats import Workflow
    from superstats.simulation import RandomChoiceContamination

    probability = Prior("normal", loc=-2, scale=0)
    process = RandomChoiceContamination(infer=infer)
    model = Model(
        JointPrior(a=1, p_contaminated=probability),
        lambda a: {"response_time": np.ones_like(a), "choice": np.zeros_like(a)},
        link_function={"p_contaminated": LinkFunction()},
        contamination=process,
        missing=None,
    )
    captured = []
    original_apply = process.apply

    def apply(data, **kwargs):
        captured.append(kwargs["probability"])
        return original_apply(data, **kwargs)

    monkeypatch.setattr(process, "apply", apply)
    workflow = Workflow.__new__(Workflow)
    workflow.model = model
    estimates = {"p_contaminated": np.full((1, 2, 1), 2.0)} if infer else {"a": np.ones((1, 2, 1))}
    result = workflow.resimulate(estimates, num_sims=1, num_steps=3, rng=0)
    expected = 1 / (1 + np.exp(-2 if infer else 2))
    np.testing.assert_allclose(captured[0], expected)
    assert result["response_time"].shape == (1, 1, 3)


def test_default_contamination_prior_does_not_mutate_callers_prior():
    from superstats.simulation import RandomChoiceContamination

    prior = JointPrior(a=1)
    model = Model(prior, simulator, contamination=RandomChoiceContamination(), missing=None)
    assert "p_contaminated" not in prior.params
    assert "p_contaminated" in model.prior.params
    assert "p_contaminated" not in model.shared_keys


def test_nuisance_transition_hyperparameters_are_excluded_from_targets():
    from superstats.simulation import RandomChoiceContamination

    model = Model(
        JointPrior(a=Prior("normal"), p_contaminated=Linear(intercept=Prior("normal"), slope=0)),
        simulator,
        contamination=RandomChoiceContamination(),
        link_function={"p_contaminated": LinkFunction()},
        missing=None,
    )
    draws = model._sample_inference_prior(2, 3)
    assert model.hyper_keys == []
    assert model.deterministic_keys == []
    assert draws["hyper_params"] == {}
    assert "p_contaminated" not in draws["hyper_param_groups"]


@pytest.mark.parametrize("batch_size", [2, 3])
def test_formula_resimulation_accepts_posterior_scalar_trajectory_shape(batch_size):
    model = Model(
        JointPrior(a_0=RandomWalk(), b_a=Prior("normal")),
        simulator,
        formula=Formula(["a = a_0 + b_a * x"]),
        context={"x": [-1, 0, 1]},
        design_context=("x",),
        missing=None,
    )
    raw = {
        "a_0": np.full((batch_size, 3, 1), 2.0),
        "b_a": np.arange(1, batch_size + 1),
    }
    result = model.simulate_from_parameters(raw, batch_size, 3)
    expected = 2 + np.arange(1, batch_size + 1)[:, None] * [-1, 0, 1]
    np.testing.assert_allclose(result["observation"], expected)


@pytest.mark.parametrize(
    "specification",
    [
        Prior("normal", loc=0, scale=0),
        RandomWalk(initial_prior=Prior("normal", loc=0, scale=0), sigma=Prior("normal", loc=0, scale=0)),
        Linear(intercept=Prior("normal", loc=0, scale=0), slope=0),
    ],
)
def test_missing_probability_and_hyperparameters_are_always_nuisance(specification):
    from superstats.simulation import RandomMissingProcess

    model = Model(
        JointPrior(a=Prior("normal"), p_missing=specification),
        simulator,
        missing=RandomMissingProcess(),
        link_function={"p_missing": LinkFunction()},
    )
    result = model.sample(2, 3)
    np.testing.assert_allclose(result["p_missing"], 0.5)
    keys = model.local_keys + model.hyper_keys + model.shared_keys + model.deterministic_keys + model.fixed_keys
    assert not any(key.startswith("p_missing") for key in keys)
    draws = model._sample_inference_prior(2, 3)
    assert "p_missing" not in draws["hyper_param_groups"]
    assert not any(key.startswith("p_missing") for group in draws.values() for key in group)


def test_missing_probability_is_sampled_and_linked_once(monkeypatch):
    from superstats.simulation import RandomMissingProcess

    probability = Prior("normal", loc=-2, scale=0)
    process = RandomMissingProcess()
    prior = JointPrior(a=1, p_missing=probability)
    model = Model(prior, simulator, missing=process, link_function={"p_missing": LinkFunction()})
    original_sample = probability.sample
    calls = []

    def sample(*args, **kwargs):
        calls.append(1)
        return original_sample(*args, **kwargs)

    monkeypatch.setattr(probability, "sample", sample)
    result = model.sample(2, 3)
    assert calls == [1]
    np.testing.assert_allclose(result["p_missing"], 1 / (1 + np.exp(2)))


def test_default_missing_prior_does_not_mutate_callers_prior():
    from superstats.defaults import DEFAULT_P_MISSING_PRIOR

    prior = JointPrior(a=1)
    model = Model(prior, simulator)
    assert "p_missing" not in prior.params
    assert model.prior.params["p_missing"] is DEFAULT_P_MISSING_PRIOR
    assert "p_missing" not in model.shared_keys


@pytest.mark.parametrize("apply_missing", [False, True])
def test_prior_push_forward_missingness_is_opt_in(monkeypatch, apply_missing):
    import superstats.simulation.model as module
    from superstats.simulation import RandomMissingProcess

    model = Model(JointPrior(a=1, p_missing=1), simulator, missing=RandomMissingProcess(missing_value=-99))
    captured = {}
    monkeypatch.setattr(module, "plot_push_forward", lambda **kwargs: captured.update(kwargs))
    kwargs = {"apply_missing": True} if apply_missing else {}
    model.plot_push_forward(batch_size=2, num_steps=3, **kwargs)
    np.testing.assert_array_equal(captured["data"]["observation"], np.full((2, 3), -99 if apply_missing else 1))


@pytest.mark.parametrize("apply_missing", [False, True])
def test_posterior_resimulation_missingness_is_opt_in(apply_missing):
    from superstats import Workflow
    from superstats.simulation import RandomMissingProcess

    model = Model(JointPrior(a=1, p_missing=1), simulator, missing=RandomMissingProcess(missing_value=-99))
    workflow = Workflow.__new__(Workflow)
    workflow.model = model
    kwargs = {"apply_missing": True} if apply_missing else {}
    result = workflow.resimulate({"a": np.ones((1, 2, 1))}, num_sims=1, num_steps=3, rng=0, **kwargs)
    np.testing.assert_array_equal(result["observation"], np.full((1, 1, 3), -99 if apply_missing else 1))
