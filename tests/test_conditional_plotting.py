"""Fixed-condition plots separate temporal evolution from changing designs."""

from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
import pytest

from superstats import Categorical, Formula, JointPrior, LinkFunction, Model, Prior, Workflow
from superstats.simulation.conditional import posterior_parameters, prior_parameters
from superstats.transition import Linear, RandomWalk


def test_existing_time_varying_prior_method_applies_the_plots_testing_links():
    from superstats.simulation import RandomChoiceContamination
    from superstats.transition import Jump, Mixture

    model = Model(
        JointPrior(
            v=RandomWalk(),
            a=Mixture([RandomWalk(), Jump()]),
            tau=RandomWalk(),
            p_contaminated=RandomWalk(),
            bias=0.5,
        ),
        lambda v, a, tau, bias: {"response_time": np.ones_like(v) + tau, "choice": np.zeros_like(v)},
        latent_link_function={
            "v": LinkFunction(bounds=(0, 6)),
            "a": LinkFunction(),
            "tau": LinkFunction(bounds=(0, 2)),
            "p_contaminated": LinkFunction(),
        },
        contamination=RandomChoiceContamination(infer=True),
        missing=None,
    )
    fig = model.plot_time_varying_prior(num_steps=6, num_trajectories=3, marginal=False, scale="model")
    assert {ax.get_title() for ax in fig.axes} == {"v", "a", "tau", "p_contaminated"}
    bounds = {"v": (0, 6), "a": (0, 1), "tau": (0, 2), "p_contaminated": (0, 1)}
    for ax in fig.axes:
        low, high = bounds[ax.get_title()]
        assert len(ax.lines) == 4
        assert all(np.all((line.get_ydata() >= low) & (line.get_ydata() <= high)) for line in ax.lines)
    plt.close(fig)
    assert not hasattr(model, "plot_prior")
    assert not hasattr(Workflow, "plot_posterior")


def test_existing_time_varying_prior_can_explicitly_show_raw_values():
    model = Model(
        JointPrior(v=RandomWalk(initial_prior=Prior("normal", loc=-2, scale=0), sigma=0, delta=0)),
        lambda v: {"observation": v},
        latent_link_function={"v": LinkFunction("exp")},
        missing=None,
    )
    for scale, expected in [("model", np.exp(-2)), ("raw", -2)]:
        fig = model.plot_time_varying_prior(num_steps=4, num_trajectories=2, marginal=False, scale=scale)
        assert all(np.allclose(line.get_ydata(), expected) for line in fig.axes[0].lines)
        plt.close(fig)


def test_existing_invariant_prior_and_joint_prior_receive_linked_samples(monkeypatch):
    from superstats.simulation import model as model_module

    model = Model(
        JointPrior(a=Prior("normal", loc=-2, scale=0)),
        lambda a: {"observation": a},
        latent_link_function={"a": LinkFunction("exp")},
        missing=None,
    )
    invariant = Mock(return_value="invariant")
    joint = Mock(return_value="joint")
    monkeypatch.setattr(model_module, "plot_time_invariant_prior", invariant)
    monkeypatch.setattr(model_module, "plot_joint_prior", joint)
    assert model.plot_time_invariant_prior(num_draws=3, scale="model") == "invariant"
    assert model.plot_joint_prior(num_draws=3, scale="model") == "joint"
    np.testing.assert_allclose(invariant.call_args.kwargs["shared_params"]["a"], np.exp(-2))
    np.testing.assert_allclose(joint.call_args.kwargs["shared_params"]["a"], np.exp(-2))


@pytest.mark.parametrize("method", ["plot_marginals", "plot_pairs", "plot_forest"])
def test_existing_invariant_posterior_methods_transform_draws_and_truth(method, monkeypatch):
    from superstats.workflow import workflow as workflow_module

    workflow = Workflow.__new__(Workflow)
    workflow.model = Model(
        JointPrior(a=Prior("normal")),
        lambda a: {"observation": a},
        latent_link_function={"a": LinkFunction("exp")},
        missing=None,
    )
    plotter = Mock(return_value="figure")
    monkeypatch.setattr(workflow_module, method, plotter)
    estimates = {"a": np.array([[[-2.0], [-1.0], [0.0]]])}
    targets = {"a": np.array([[-1.0]])}
    assert getattr(workflow, method)(estimates=estimates, targets=targets, scale="model") == "figure"
    np.testing.assert_allclose(plotter.call_args.kwargs["estimates"]["a"], np.exp(estimates["a"]))
    np.testing.assert_allclose(plotter.call_args.kwargs["targets"]["a"], np.exp(targets["a"]))


@pytest.mark.parametrize("method", ["plot_marginals", "plot_pairs", "plot_forest"])
def test_dynamic_marginals_require_a_time_selection_and_keep_condition_pairing(method, monkeypatch):
    from superstats.workflow import workflow as workflow_module

    workflow = Workflow.__new__(Workflow)
    workflow.model = regression_model()
    estimates = {"v_0": np.arange(12, dtype=float).reshape(1, 3, 4, 1)}
    with pytest.raises(ValueError, match="time_step"):
        getattr(workflow, method)(estimates, variable_keys=["v"])
    plotter = Mock(return_value="figure")
    monkeypatch.setattr(workflow_module, method, plotter)
    assert getattr(workflow, method)(estimates, variable_keys=["v"], time_step=2, scale="model") == "figure"
    values = plotter.call_args.kwargs["estimates"]["v"]
    np.testing.assert_array_equal(values, [[[2, 4], [6, 8], [10, 12]]])
    assert plotter.call_args.kwargs["mixture_names"]["v"] == ["difficulty=0", "difficulty=1"]


@pytest.mark.parametrize("scale", ["both", "model"])
def test_condition_legend_in_marginals_is_disabled(scale):
    workflow = Workflow.__new__(Workflow)
    workflow.model = regression_model()
    estimates = {"v_0": np.arange(12, dtype=float).reshape(1, 3, 4, 1)}
    fig = workflow.plot_marginals(estimates, variable_keys=["v"], time_step=2, scale=scale)
    assert not fig.legends
    assert all(ax.get_legend() is None for ax in fig.axes)
    plt.close(fig)


def test_multiple_dummy_conditions_get_distinct_trajectory_colors():
    model = Model(
        JointPrior(v0=RandomWalk(sigma=0, delta=0)),
        lambda v: {"observation": v},
        formula=Formula(["v = v0 + a + 2 * b + 4 * c"]),
        context_simulator={name: [0, 1] for name in ("a", "b", "c")},
        design_context=("a", "b", "c"),
        missing=None,
    )
    fig = model.plot_time_varying_prior(num_steps=2, num_trajectories=1, marginal=False, scale="model")
    colors = {tuple(line.get_color()) for line in fig.axes[0].lines if line.get_linewidth() == 2.5}
    assert len(colors) == 8
    plt.close(fig)


def regression_model():
    return Model(
        JointPrior(
            v_0=RandomWalk(initial_prior=Prior("normal", loc=1, scale=0.2), sigma=0.05, delta=0),
            b_v=2,
        ),
        lambda v: {"observation": v},
        formula=Formula(["v = v_0 + b_v * difficulty"]),
        context_simulator={"difficulty": [0, 1, 0, 1]},
        design_context=("difficulty",),
        missing=None,
    )


def test_prior_reuses_each_random_walk_draw_across_fixed_dummy_conditions():
    model = regression_model()
    model.prior.sample = Mock(wraps=model.prior.sample)
    model.simulator = Mock(side_effect=AssertionError("Plotting must not simulate observations"))
    draws = prior_parameters(model, "v", num_trajectories=5, num_steps=12)

    assert draws.conditions == [{"difficulty": 0}, {"difficulty": 1}]
    assert draws.parameters["v"].shape == (1, 5, 2, 12)
    np.testing.assert_allclose(draws.parameters["v"][0, :, 1] - draws.parameters["v"][0, :, 0], 2)
    np.testing.assert_allclose(np.diff(draws.parameters["v"][0, :, 1]), np.diff(draws.parameters["v"][0, :, 0]))
    model.prior.sample.assert_called_once_with(batch_size=5, num_steps=12)
    model.simulator.assert_not_called()


def test_prior_plot_preserves_individual_draw_count_and_average_per_condition():
    fig = regression_model().plot_time_varying_prior(
        variable_keys=["v"], num_trajectories=5, num_steps=8, marginal=False, scale="model"
    )
    ax = fig.axes[0]
    assert len(ax.lines) == 12  # five individual trajectories + their average, for both conditions
    assert len(ax.collections) == 0  # prior draws, rather than uncertainty ribbons
    assert {line.get_label() for line in ax.lines if line.get_linewidth() == 2.5} == {"difficulty=0", "difficulty=1"}
    first, second = ax.lines[:6], ax.lines[6:]
    for a, b in zip(first, second):
        np.testing.assert_allclose(b.get_ydata() - a.get_ydata(), 2)
    assert first[0].get_color() != second[0].get_color()
    plt.close(fig)


def test_links_and_ordered_formulas_use_the_simulation_order_for_each_draw():
    model = Model(
        JointPrior(base=Linear(intercept=-1, slope=1, normalize_steps=False), slope=2),
        lambda v: {"observation": v},
        formula=Formula(["offset = slope * x", "v = base + offset"]),
        latent_link_function={"slope": LinkFunction("exp")},
        formula_link_function={"v": LinkFunction(bounds=(0.2, 4))},
        context_simulator={"x": [0, 1]},
        design_context=("x",),
        missing=None,
    )
    draws = prior_parameters(model, "v", num_steps=3, num_trajectories=2)
    for condition in (0, 1):
        expected = LinkFunction(bounds=(0.2, 4))(np.array([-1, 0, 1]) + np.exp(2) * condition)
        np.testing.assert_allclose(draws.parameters["v"][0, :, condition], np.tile(expected, (2, 1)))


def test_continuous_levels_are_constant_over_time_and_based_on_reference_data():
    model = regression_model()
    draws = prior_parameters(model, "v", num_steps=6, context={"difficulty": [-2, 0, 2]})
    sd = np.std([-2, 0, 2])
    np.testing.assert_allclose([row["difficulty"] for row in draws.conditions], [-sd, 0, sd])
    np.testing.assert_allclose(draws.parameters["v"][:, :, 2] - draws.parameters["v"][:, :, 1], 2 * sd)


def test_masked_reference_values_do_not_turn_a_dummy_into_a_continuous_predictor():
    draws = prior_parameters(
        regression_model(),
        "v",
        num_steps=6,
        context={"difficulty": [[0, -999, 1]], "missing_mask": [[False, True, False]]},
    )
    assert draws.conditions == [{"difficulty": 0}, {"difficulty": 1}]


def test_explicit_grids_and_selected_combinations_support_categorical_interactions():
    model = Model(
        JointPrior(base=Linear(intercept=1, slope=1), low=2, high=4, interaction=3),
        lambda v: {"observation": v},
        formula=Formula(
            ["v = base + C(group, low=low, high=high) + interaction * x * C(group, low=1, high=2)"],
            categorical={"group": Categorical(["control", "low", "high"], reference="control")},
        ),
        context_simulator={"group": ["control"], "x": [0]},
        design_context=("group", "x"),
        missing=None,
    )
    draws = prior_parameters(
        model, "v", num_steps=3, num_trajectories=1, conditions={"group": ["low", "high"], "x": [0, 1]}
    )
    np.testing.assert_allclose(draws.parameters["v"][0, 0], [[3, 3.5, 4], [6, 6.5, 7], [5, 5.5, 6], [11, 11.5, 12]])
    selected = prior_parameters(
        model,
        "v",
        num_steps=3,
        num_trajectories=1,
        conditions=[{"group": "control", "x": 0}, {"group": "high", "x": 1}],
    )
    assert len(selected.conditions) == 2
    np.testing.assert_allclose(selected.parameters["v"][0, 0], [[1, 1.5, 2], [11, 11.5, 12]])


def test_only_context_dependencies_of_the_selected_parameter_are_varied():
    model = Model(
        JointPrior(base=1),
        lambda v, a: {"observation": v + a},
        formula=Formula(["offset = base + x", "v = offset", "a = base + y"]),
        context_simulator={"x": [0, 1], "y": [0, 1]},
        design_context=("x", "y"),
        missing=None,
    )
    draws = prior_parameters(model, "v", num_steps=4)
    assert draws.conditions == [{"x": 0, "y": 0}, {"x": 1, "y": 0}]


def test_large_automatic_grids_require_selected_conditions():
    model = Model(
        JointPrior(base=1),
        lambda v: {"observation": v},
        formula=Formula(["v = base + x * y * z"]),
        context_simulator={key: [-1, 0, 1] for key in ("x", "y", "z")},
        design_context=("x", "y", "z"),
        missing=None,
    )
    with pytest.raises(ValueError, match="27 groups"):
        prior_parameters(model, "v")
    draws = prior_parameters(model, "v", conditions=[{"x": -1, "y": 0, "z": 1}])
    assert len(draws.conditions) == 1


def test_posterior_draw_pairing_and_dataset_selection_are_preserved_without_prior_sampling():
    model = regression_model()
    model.prior.params["b_v"] = Prior("normal")
    model.prior.sample = Mock(side_effect=AssertionError("Posterior plotting must not draw from the prior"))
    base = np.arange(24).reshape(2, 3, 4, 1).astype(float)
    slope = np.array([[[1], [2], [3]], [[4], [5], [6]]])
    estimates = {"v_0": base.copy(), "b_v": slope.copy()}
    draws = posterior_parameters(model, estimates, "v", conditions={"difficulty": [0, 1]}, data_idx=[1, 0])
    np.testing.assert_array_equal(draws.data_indices, [1, 0])
    np.testing.assert_array_equal(draws.parameters["v"][:, :, 0], base[[1, 0], ..., 0])
    np.testing.assert_array_equal(draws.parameters["v"][:, :, 1], base[[1, 0], ..., 0] + slope[[1, 0]])
    np.testing.assert_array_equal(estimates["v_0"], base)
    np.testing.assert_array_equal(estimates["b_v"], slope)
    model.prior.sample.assert_not_called()


def test_posterior_reconstructs_deterministic_trajectories_from_coefficient_draws():
    model = Model(
        JointPrior(base=Linear(intercept=Prior("normal"), slope=2), slope=1),
        lambda v: {"observation": v},
        formula=Formula(["v = base + slope * x"]),
        context_simulator={"x": [0, 1, 0]},
        design_context=("x",),
        missing=None,
        formula_link_function={"v": LinkFunction("softplus")},
    )
    model.prior.sample = Mock(side_effect=AssertionError("No new prior draws"))
    draws = posterior_parameters(
        model,
        {"base_intercept": np.array([[[1], [2]]])},
        "v",
        conditions={"x": [0, 1]},
        num_steps=3,
    )
    np.testing.assert_allclose(draws.parameters["v"][0, :, 0], np.logaddexp(0, [[1, 2, 3], [2, 3, 4]]))
    np.testing.assert_allclose(draws.parameters["v"][0, :, 1], np.logaddexp(0, [[2, 3, 4], [3, 4, 5]]))


def test_posterior_plot_uses_intervals_and_can_overlay_individuals():
    workflow = Workflow.__new__(Workflow)
    workflow.model = regression_model()
    estimates = {"v_0": np.arange(12).reshape(1, 3, 4, 1)}
    fig = workflow.plot_time_varying_posterior(
        estimates, variable_keys=["v"], num_trajectories=2, marginal=False, scale="model"
    )
    ax = fig.axes[0]
    assert len(ax.lines) == 6
    assert len(ax.collections) == 4
    means = [line.get_ydata() for line in ax.lines if line.get_linewidth() == 2.5]
    np.testing.assert_array_equal(means[0], [4, 5, 6, 7])
    np.testing.assert_array_equal(means[1], [6, 7, 8, 9])
    plt.close(fig)


def test_linked_parameters_without_a_formula_or_context_are_plottable():
    model = Model(
        JointPrior(a=Prior("normal", loc=-2, scale=0)),
        lambda a: {"observation": a},
        latent_link_function={"a": LinkFunction("exp")},
        missing=None,
    )
    draws = prior_parameters(model, num_steps=3, num_trajectories=2)
    np.testing.assert_allclose(draws.parameters["a"], np.exp(-2))
    assert draws.conditions == [{}]
    fig = model.plot_time_varying_prior(variable_keys=["a"], num_steps=3, num_trajectories=2, scale="model")
    assert len(fig.axes) == 3  # hidden layout axis, trajectory, marginal
    plt.close(fig)


@pytest.mark.parametrize(
    "conditions,error",
    [
        ({"typo": 0}, ValueError),
        ({"difficulty": []}, ValueError),
        ({"difficulty": [[0, 1]]}, ValueError),
        ([{"difficulty": [0, 1]}], ValueError),
        ([], ValueError),
        ([1], ValueError),
        ("difficulty", TypeError),
        ({"difficulty": float("nan")}, ValueError),
    ],
)
def test_invalid_conditions_are_rejected(conditions, error):
    with pytest.raises(error):
        prior_parameters(regression_model(), "v", conditions=conditions)


def test_posterior_rejects_mismatched_draw_shapes_and_extrapolation():
    model = regression_model()
    estimates = {"v_0": np.zeros((1, 3, 4, 1))}
    with pytest.raises(ValueError, match="not extrapolated"):
        posterior_parameters(model, estimates, "v", num_steps=5)
    with pytest.raises(ValueError, match="axes"):
        posterior_parameters(model, {**estimates, "other": np.zeros((1, 2, 1))}, "v")
    with pytest.raises(KeyError, match="v_0"):
        posterior_parameters(model, {"other": np.zeros((1, 3, 1))}, "v")


@pytest.mark.parametrize("dynamic", ["b0", "b1"])
def test_both_prior_views_keep_coefficients_and_implied_curves_paired(dynamic):
    model = Model(
        JointPrior(**{name: RandomWalk() if name == dynamic else Prior("normal") for name in ("b0", "b1")}),
        lambda mu: {"observation": mu},
        formula=Formula(["mu = b0 + b1 * condition"]),
        latent_link_function={dynamic: LinkFunction("exp")},
        formula_link_function={"mu": LinkFunction("softplus")},
        design_context=("condition",),
        context_simulator={"condition": [0, 1]},
        missing=None,
    )
    model.prior.sample = Mock(wraps=model.prior.sample)
    fig = model.plot_time_varying_prior(num_steps=5, num_trajectories=3, marginal=False)
    panels = {ax.get_title(): ax for ax in fig.axes if ax.get_visible()}
    assert set(panels) == {f"{dynamic} [raw]", "mu [implied]"}
    raw_lines = panels[f"{dynamic} [raw]"].lines
    implied_lines = panels["mu [implied]"].lines
    assert len(raw_lines) == 4
    assert len(implied_lines) == 8
    # Differences on the formula's input scale recover each draw of the linked slope.
    implied_0 = np.stack([line.get_ydata() for line in implied_lines[:3]])
    implied_1 = np.stack([line.get_ydata() for line in implied_lines[4:7]])

    def inverse_softplus(value):
        return np.log(np.expm1(value))

    if dynamic == "b1":
        np.testing.assert_allclose(
            inverse_softplus(implied_1) - inverse_softplus(implied_0),
            np.exp(np.stack([line.get_ydata() for line in raw_lines[:3]])),
        )
    else:
        np.testing.assert_allclose(
            inverse_softplus(implied_0), np.exp(np.stack([line.get_ydata() for line in raw_lines[:3]]))
        )
    model.prior.sample.assert_called_once_with(batch_size=3, num_steps=5)
    assert all(ax.get_legend() is None for ax in fig.axes)
    plt.close(fig)


@pytest.mark.parametrize("method", ["plot_time_invariant_prior", "plot_joint_prior"])
def test_both_invariant_prior_views_share_one_sample(method, monkeypatch):
    from superstats.simulation import model as model_module

    model = Model(
        JointPrior(a=Prior("normal")),
        lambda a: {"observation": a},
        latent_link_function={"a": LinkFunction("exp")},
        missing=None,
    )
    model.prior.sample = Mock(wraps=model.prior.sample)
    plotter = Mock(return_value="figure")
    monkeypatch.setattr(model_module, method, plotter)
    assert getattr(model, method)(num_draws=7) == "figure"
    values = plotter.call_args.kwargs["shared_params"]
    assert set(values) == {"a [raw]", "a [implied]"}
    np.testing.assert_allclose(values["a [implied]"], np.exp(values["a [raw]"]))
    assert model.prior.sample.call_count == 1


@pytest.mark.parametrize("method", ["plot_marginals", "plot_pairs", "plot_forest"])
def test_both_invariant_posterior_views_transform_paired_estimates_and_targets(method, monkeypatch):
    from superstats.workflow import workflow as workflow_module

    workflow = Workflow.__new__(Workflow)
    workflow.model = Model(
        JointPrior(a=Prior("normal")),
        lambda a: {"observation": a},
        latent_link_function={"a": LinkFunction("exp")},
        missing=None,
    )
    workflow.model.prior.sample = Mock(side_effect=AssertionError("Never resample a posterior"))
    plotter = Mock(return_value="figure")
    monkeypatch.setattr(workflow_module, method, plotter)
    estimates = {"a": np.array([[[-2.0], [-1.0], [0.0]]])}
    targets = {"a": np.array([[-1.0]])}
    assert getattr(workflow, method)(estimates, targets) == "figure"
    result = plotter.call_args.kwargs
    assert result["variable_keys"] == ["a [raw]", "a [implied]"]
    np.testing.assert_array_equal(result["estimates"]["a [raw]"], estimates["a"])
    np.testing.assert_allclose(result["estimates"]["a [implied]"], np.exp(estimates["a"]))
    np.testing.assert_array_equal(result["targets"]["a [raw]"], targets["a"])
    np.testing.assert_allclose(result["targets"]["a [implied]"], np.exp(targets["a"]))
    workflow.model.prior.sample.assert_not_called()


def test_both_posterior_trajectory_views_have_raw_truth_and_fixed_condition_truth():
    workflow = Workflow.__new__(Workflow)
    workflow.model = regression_model()
    estimates = {"v_0": np.arange(12).reshape(1, 3, 4, 1)}
    targets = {"v_0": np.arange(4).reshape(1, 4, 1)}
    fig = workflow.plot_time_varying_posterior(
        estimates,
        targets,
        marginal=False,
        uncertainty_fun=None,
        num_trajectories=3,
        variable_names=["Drift"],
    )
    axes = {ax.get_title(): ax for ax in fig.axes if ax.get_visible()}
    assert set(axes) == {"v_0 [raw] — Dataset 0", "Drift [implied] — Dataset 0"}
    raw = axes["v_0 [raw] — Dataset 0"].lines
    implied = axes["Drift [implied] — Dataset 0"].lines
    assert len(raw) == 5  # three draws, median, truth
    assert len(implied) == 10  # same for each condition
    for draw in range(5):
        np.testing.assert_array_equal(raw[draw].get_ydata(), implied[draw].get_ydata())
        np.testing.assert_array_equal(implied[draw + 5].get_ydata(), raw[draw].get_ydata() + 2)
    assert all(ax.get_legend() is None for ax in fig.axes)
    assert not any("difficulty" in label.get_text() for legend in fig.legends for label in legend.get_texts())
    plt.close(fig)


@pytest.mark.parametrize("marginal", [False, True])
@pytest.mark.parametrize("aggregation", [None, np.mean])
def test_posterior_groups_each_parameters_views_and_keeps_teal_legend_on_one_line(marginal, aggregation):
    from matplotlib.colors import to_rgba

    from superstats.defaults import BASE_COLOR

    workflow = Workflow.__new__(Workflow)
    workflow.model = Model(
        JointPrior(v0=RandomWalk(), bv=RandomWalk(), a0=RandomWalk()),
        lambda v, a: {"observation": v + a},
        formula=Formula(["v = v0 + bv * x", "a = a0"]),
        latent_link_function={"a0": LinkFunction("exp")},
        design_context=("x",),
        context_simulator={"x": [0, 1]},
        missing=None,
    )
    values = np.arange(24, dtype=float).reshape(2, 3, 4, 1) / 20
    estimates = {"v0": values, "bv": values + 1, "a0": values + 2}
    targets = {name: samples[:, 0] for name, samples in estimates.items()}
    fig = workflow.plot_time_varying_posterior(
        estimates, targets, marginal=marginal, aggregation=aggregation, data_idx=[1, 0]
    )
    axes = [ax for ax in fig.axes if ax.get_visible() and ax.get_title()]
    dataset_indices = [None] if aggregation is not None else [1, 0]
    expected_row = 0
    for group in [["v0 [raw]", "bv [raw]", "v [implied]"], ["a0 [raw]", "a [implied]"]]:
        for dataset in dataset_indices:
            for column, name in enumerate(group):
                title = name if dataset is None else f"{name} — Dataset {dataset}"
                ax = next(ax for ax in axes if ax.get_title() == title)
                spec = ax.get_subplotspec().get_topmost_subplotspec()
                assert (spec.rowspan.start, spec.colspan.start) == (expected_row, column)
                if name == "v0 [raw]":
                    expected = (
                        values[..., 0].mean(axis=0).mean(axis=0)
                        if dataset is None
                        else values[dataset, ..., 0].mean(axis=0)
                    )
                    np.testing.assert_allclose(ax.lines[0].get_ydata(), expected)
            expected_row += 1

    fig.canvas.draw()
    legend = fig.legends[0]
    assert [text.get_text() for text in legend.get_texts()] == ["Median", "Target", "95% HDI", "65% HDI"]
    center, target, outer, inner = legend.legend_handles
    assert to_rgba(center.get_color()) == to_rgba(BASE_COLOR)
    assert to_rgba(target.get_color()) == to_rgba("black")
    assert outer.get_facecolor() == to_rgba(BASE_COLOR, 0.25)
    assert inner.get_facecolor() == to_rgba(BASE_COLOR, 0.5)
    assert np.ptp([text.get_window_extent().y0 for text in legend.get_texts()]) < 1
    plt.close(fig)


def test_single_column_posterior_legend_fits_with_individual_draws_and_custom_color():
    workflow = Workflow.__new__(Workflow)
    workflow.model = regression_model()
    estimates = {"v_0": np.arange(12).reshape(1, 3, 4, 1)}
    targets = {"v_0": np.arange(4).reshape(1, 4, 1)}
    fig = workflow.plot_time_varying_posterior(
        estimates,
        targets,
        scale="model",
        conditions={"difficulty": [0]},
        marginal=False,
        num_trajectories=1,
        color="#008080",
    )
    fig.canvas.draw()
    legend = fig.legends[0]
    box = legend.get_window_extent()
    assert 0 <= box.x0 < box.x1 <= fig.bbox.width
    assert np.ptp([text.get_window_extent().y0 for text in legend.get_texts()]) < 1
    assert legend.legend_handles[0].get_color() == "#008080"
    plt.close(fig)


def test_both_joint_prior_shows_hyperparameters_only_with_the_raw_process(monkeypatch):
    from superstats.simulation import model as model_module

    model = regression_model()
    model.prior.params["v_0"] = RandomWalk(sigma=Prior("halfnormal"))
    plotter = Mock(return_value="figure")
    monkeypatch.setattr(model_module, "plot_joint_prior", plotter)
    model.plot_joint_prior(num_draws=7, num_trajectories=2, num_steps=5)
    result = plotter.call_args.kwargs
    assert set(result["local_params"]) == {"v_0 [raw]", "v [implied]"}
    assert all(value.shape[0] == 2 for value in result["local_params"].values())
    assert result["hyper_param_groups"]["v [implied]"] == []
    assert result["hyper_param_groups"]["v_0 [raw]"] == ["v_0_sigma"]
    assert list(result["hyper_params"]) == ["v_0_sigma"]
    assert result["hyper_params"]["v_0_sigma"].shape[0] == 7


def test_identical_raw_and_implied_parameters_use_a_single_panel():
    model = Model(JointPrior(mu=RandomWalk()), lambda mu: {"observation": mu}, missing=None)
    fig = model.plot_time_varying_prior(num_steps=4, num_trajectories=2, marginal=False)
    assert [ax.get_title() for ax in fig.axes] == ["mu"]
    plt.close(fig)


@pytest.mark.parametrize("method", ["plot_marginals", "plot_pairs", "plot_forest"])
def test_both_dynamic_snapshots_reduce_raw_and_implied_without_breaking_pairing(method, monkeypatch):
    from superstats.workflow import workflow as workflow_module

    workflow = Workflow.__new__(Workflow)
    workflow.model = regression_model()
    plotter = Mock(return_value="figure")
    monkeypatch.setattr(workflow_module, method, plotter)
    estimates = {"v_0": np.arange(12).reshape(1, 3, 4, 1)}
    getattr(workflow, method)(estimates, variable_keys=["v"], time_step=2)
    values = plotter.call_args.kwargs["estimates"]
    np.testing.assert_array_equal(values["v_0 [raw]"], [[[2], [6], [10]]])
    np.testing.assert_array_equal(values["v [implied]"], [[[2, 4], [6, 8], [10, 12]]])


@pytest.mark.parametrize(
    "method", ["plot_time_invariant_prior", "plot_joint_prior", "plot_marginals", "plot_pairs", "plot_forest"]
)
def test_both_vector_parameter_views_preserve_components_and_pairing(method, monkeypatch):
    from superstats.simulation import model as model_module
    from superstats.workflow import workflow as workflow_module

    model = Model(
        JointPrior(weights=Prior("dirichlet", alpha=[1, 1, 1])),
        lambda weights: {"observation": weights.sum(axis=-1)},
        latent_link_function={"weights": LinkFunction("exp")},
        missing=None,
    )
    plotter = Mock(return_value="figure")
    if method.endswith("prior"):
        monkeypatch.setattr(model_module, method, plotter)
        getattr(model, method)(num_draws=7)
        result = plotter.call_args.kwargs["shared_params"]
    else:
        workflow = Workflow.__new__(Workflow)
        workflow.model = model
        monkeypatch.setattr(workflow_module, method, plotter)
        estimates = {"weights": np.arange(21).reshape(1, 7, 3) / 21}
        getattr(workflow, method)(estimates)
        result = plotter.call_args.kwargs["estimates"]
    assert set(result) == {"weights [raw]", "weights [implied]"}
    np.testing.assert_allclose(result["weights [implied]"], np.exp(result["weights [raw]"]))
    assert result["weights [raw]"].shape[-1] == 3


@pytest.mark.parametrize("scale", ["both", "model"])
@pytest.mark.parametrize("method", ["plot_time_varying_prior", "plot_time_invariant_prior", "plot_joint_prior"])
def test_prior_condition_legends_are_hidden_for_each_layout(method, scale):
    model = Model(
        JointPrior(base=RandomWalk(), intercept=Prior("normal")),
        lambda mu, a: {"observation": mu + a},
        formula=Formula(["mu = base + x", "a = intercept + x"]),
        design_context=("x",),
        context_simulator={"x": [0, 1]},
        missing=None,
    )
    options = {"num_draws": 7} if method != "plot_time_varying_prior" else {"num_trajectories": 2}
    fig = getattr(model, method)(num_steps=4, scale=scale, **options)
    assert all(ax.get_legend() is None for ax in fig.axes)
    plt.close(fig)


def test_condition_legends_stay_hidden_with_custom_marginal_names():
    workflow = Workflow.__new__(Workflow)
    workflow.model = regression_model()
    estimates = {"v_0": np.arange(12).reshape(1, 3, 4, 1)}
    fig = workflow.plot_marginals(estimates, variable_keys=["v"], variable_names=["Drift"], time_step=2)
    titles = {ax.get_title() for ax in fig.axes}
    assert "Drift [implied]" in titles
    assert not fig.legends
    plt.close(fig)


@pytest.mark.parametrize("scale", ["both", "model", "raw"])
@pytest.mark.parametrize("marginal", [False, True])
def test_joint_prior_groups_coefficients_under_observation_parameter_rows(scale, marginal):
    model = Model(
        JointPrior(
            v0=RandomWalk(sigma=Prior("halfnormal")),
            b_v=RandomWalk(sigma=Prior("halfnormal")),
            a0=RandomWalk(sigma=Prior("halfnormal")),
            b_a=Prior("normal"),
        ),
        lambda v, a: {"observation": v + a},
        formula=Formula(["v = v0 + b_v * x", "a = a0 + b_a * x"]),
        latent_link_function={"v0": LinkFunction("exp")},
        formula_link_function={"a": LinkFunction(bounds=(1, 4))},
        design_context=("x",),
        context_simulator={"x": [0, 1]},
        missing=None,
    )
    model.prior.sample = Mock(wraps=model.prior.sample)
    fig = model.plot_joint_prior(num_draws=7, num_trajectories=2, num_steps=5, scale=scale, marginal=marginal)
    axes = [ax for ax in fig.axes if ax.axison]
    row_labels = [
        (text.get_text(), ax.get_subplotspec().get_topmost_subplotspec().rowspan.start)
        for ax in fig.axes
        for text in ax.texts
    ]
    assert row_labels == [("v", 0), ("a", 1)]
    assert {ax.get_subplotspec().get_topmost_subplotspec().rowspan.start for ax in axes} == {0, 1}
    titles = [ax.get_title() for ax in axes]
    assert titles.count("Raw") == (4 if scale != "model" else 0)
    assert titles.count("Implied") == (2 if scale != "raw" else 0)
    assert all(ax.get_legend() is None for ax in axes)
    if scale == "both":
        v_raw = {ax.get_xlabel(): ax for ax in axes if ax.get_title() == "Raw" and ax.get_xlabel() in {"v0", "b_v"}}
        v_implied = next(
            ax
            for ax in axes
            if ax.get_title() == "Implied" and ax.get_subplotspec().get_topmost_subplotspec().rowspan.start == 0
        )
        for draw in range(2):
            np.testing.assert_allclose(v_implied.lines[draw].get_ydata(), np.exp(v_raw["v0"].lines[draw].get_ydata()))
            np.testing.assert_allclose(
                v_implied.lines[draw + 3].get_ydata(),
                np.exp(v_raw["v0"].lines[draw].get_ydata()) + v_raw["b_v"].lines[draw].get_ydata(),
            )
        a_implied = next(
            ax
            for ax in axes
            if ax.get_title() == "Implied" and ax.get_subplotspec().get_topmost_subplotspec().rowspan.start == 1
        )
        assert a_implied.get_ylim() == (1, 4)
    model.prior.sample.assert_called_once_with(batch_size=7, num_steps=5)
    plt.close(fig)


def test_joint_prior_groups_stationary_raw_and_implied_distributions_in_one_row():
    model = Model(
        JointPrior(a=Prior("normal")),
        lambda a: {"observation": a},
        latent_link_function={"a": LinkFunction("exp")},
        missing=None,
    )
    fig = model.plot_joint_prior(num_draws=10)
    axes = [ax for ax in fig.axes if ax.axison]
    assert [ax.get_title() for ax in axes] == ["Raw", "Implied"]
    assert [text.get_text() for ax in fig.axes for text in ax.texts] == ["a"]
    assert {ax.get_subplotspec().get_topmost_subplotspec().rowspan.start for ax in axes} == {0}
    plt.close(fig)


def test_joint_prior_identical_views_share_one_panel_in_the_parameter_row():
    model = Model(JointPrior(mu=RandomWalk()), lambda mu: {"observation": mu}, missing=None)
    fig = model.plot_joint_prior(num_draws=5, num_trajectories=2, num_steps=4, marginal=False)
    axes = [ax for ax in fig.axes if ax.axison]
    assert [ax.get_title() for ax in axes].count("Raw / Implied") == 1
    assert [text.get_text() for ax in fig.axes for text in ax.texts] == ["mu"]
    plt.close(fig)


def test_joint_prior_shared_process_hyperparameters_appear_only_once_across_target_rows(monkeypatch):
    from superstats.simulation import model as model_module

    model = Model(
        JointPrior(base=RandomWalk(sigma=Prior("halfnormal"))),
        lambda v, a: {"observation": v + a},
        formula=Formula(["v = base", "a = base + 1"]),
        missing=None,
    )
    plotter = Mock(return_value="figure")
    monkeypatch.setattr(model_module, "plot_joint_prior", plotter)
    model.plot_joint_prior(num_draws=5, num_trajectories=2, num_steps=4)
    groups = plotter.call_args.kwargs["parameter_groups"]
    assert list(groups) == ["v", "a"]
    assert groups["v"]["hyper"] == ["base_sigma"]
    assert groups["a"]["hyper"] == []
    assert groups["v"]["raw"] == groups["a"]["raw"] == ["base [raw]"]
