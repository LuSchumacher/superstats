import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import superstats.diagnostics.plots.calibration as calibration_module
import superstats.diagnostics.plots.recovery as recovery_module
from superstats.diagnostics.plots import (
    plot_joint_prior,
    plot_posterior_resimulation,
    plot_push_forward,
    plot_time_invariant_prior,
    plot_time_varying_posterior,
    plot_time_varying_verification,
)


@pytest.mark.parametrize(
    ("data", "kwargs", "error", "message"),
    [
        (np.ones((2, 3)), {}, TypeError, "mapping"),
        ({"value": np.ones((2, 3))}, {"data_dim": 2}, ValueError, "out of range"),
        ({"value": np.ones((2, 3))}, {"data_dim": "missing"}, KeyError, "not found"),
        ({"value": np.ones((2, 3, 1))}, {}, ValueError, "must have shape"),
        ({"value": np.ones((2, 3))}, {"dist_type": "violin"}, ValueError, "dist_type"),
        (
            {"value": np.ones((2, 3))},
            {"num_cols": 0, "uncertainty_fun": None},
            ValueError,
            "num_cols",
        ),
    ],
    ids=["not-mapping", "index", "name", "shape", "dist-type", "columns"],
)
def test_plot_push_forward_validates_inputs(data, kwargs, error, message):
    with pytest.raises(error, match=message):
        plot_push_forward(data, **kwargs)


def test_plot_push_forward_warns_when_uncertainty_is_not_applicable():
    data = {"value": np.arange(12, dtype=float).reshape(3, 4)}

    with pytest.warns(UserWarning, match="requires aggregation"):
        time_fig = plot_push_forward(data, kind="time_series", marginal=False)

    with pytest.warns(UserWarning, match="not supported"):
        dist_fig = plot_push_forward(data, kind="dist", aggregation=np.mean)

    plt.close(time_fig)
    plt.close(dist_fig)


def _two_bound_uncertainty(values):
    return values.min(axis=0), values.max(axis=0)


def _three_bound_uncertainty(values):
    return values.mean(axis=0), values.min(axis=0), values.max(axis=0)


@pytest.mark.parametrize(
    "uncertainty_fun",
    ["ci", "mad", "hdi", _two_bound_uncertainty, _three_bound_uncertainty],
    ids=["confidence-interval", "mad", "hdi", "callable-two", "callable-three"],
)
def test_plot_push_forward_supports_uncertainty_modes(uncertainty_fun):
    values = np.random.default_rng(10).normal(size=(20, 6))

    fig = plot_push_forward(
        {"value": values},
        kind="time_series",
        aggregation=np.mean,
        uncertainty_fun=uncertainty_fun,
        marginal=False,
    )

    expected_ribbons = 1 if callable(uncertainty_fun) else 2
    assert len(fig.axes[0].collections) == expected_ribbons
    assert any(
        "Uncertainty" in text.get_text() or "%" in text.get_text() or "MAD" in text.get_text()
        for text in fig.legends[0].get_texts()
    )
    plt.close(fig)


def test_all_time_series_diagnostics_draw_nested_named_uncertainty_ribbons():
    rng = np.random.default_rng(12)
    trajectories = rng.normal(size=(3, 20, 6))
    targets = rng.normal(size=(3, 6))

    resimulation_fig = plot_posterior_resimulation(
        {"value": trajectories},
        {"value": targets},
        aggregation=np.mean,
        uncertainty_fun="ci",
        marginal=False,
    )
    posterior_fig = plot_time_varying_posterior(
        {"value": trajectories[..., None]},
        aggregation=np.mean,
        uncertainty_fun="mad",
        marginal=False,
    )

    assert len(resimulation_fig.axes[0].collections) == 2
    assert len(posterior_fig.axes[0].collections) == 2
    plt.close(resimulation_fig)
    plt.close(posterior_fig)


@pytest.mark.parametrize(
    "uncertainty_fun",
    ["unknown", lambda values: (values.mean(axis=0),)],
    ids=["unknown-name", "invalid-callable-result"],
)
def test_plot_push_forward_rejects_invalid_uncertainty(uncertainty_fun):
    with pytest.raises(ValueError, match="uncertainty"):
        plot_push_forward(
            {"value": np.ones((4, 3))},
            kind="time_series",
            aggregation=np.mean,
            uncertainty_fun=uncertainty_fun,
            marginal=False,
        )


def test_plot_push_forward_discrete_aggregate_has_marginal_and_spaghetti_legend():
    values = np.array(
        [
            [0, 0, 1, 1],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
        ]
    )

    fig = plot_push_forward(
        {"choice": values},
        kind="time_series",
        aggregation=np.median,
        uncertainty_fun=None,
        marginal=True,
        spaghetti=True,
    )

    visible_axes = [ax for ax in fig.axes if ax.axison]
    assert visible_axes[0].get_yticks().tolist() == [0, 1]
    assert any(ax.patches for ax in fig.axes)
    assert "Individual" in [text.get_text() for text in fig.legends[0].get_texts()]
    plt.close(fig)


def test_plot_push_forward_discrete_aggregate_distribution_is_normalized():
    values = np.array([[0, 0, 1, 1], [0, 1, 1, 1], [0, 0, 0, 1]])

    fig = plot_push_forward(
        {"choice": values},
        kind="dist",
        aggregation=np.mean,
        uncertainty_fun=None,
    )

    assert sum(patch.get_height() for patch in fig.axes[0].patches) == pytest.approx(1.0)
    plt.close(fig)


@pytest.mark.parametrize(
    "values",
    [
        np.arange(15, dtype=float).reshape(3, 5) / 10,
        np.array([[0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]]),
    ],
    ids=["continuous", "discrete"],
)
def test_plot_push_forward_individual_time_series_supports_marginals(values):
    fig = plot_push_forward(
        {"value": values},
        kind="time_series",
        aggregation=None,
        uncertainty_fun=None,
        marginal=True,
        num_cols=2,
    )

    assert any(not ax.axison for ax in fig.axes)
    assert len([ax for ax in fig.axes if ax.axison]) == len(values)
    assert len(fig.axes) > len(values)
    plt.close(fig)


@pytest.mark.parametrize(
    ("prediction", "empirical", "kwargs", "error", "message"),
    [
        (np.ones((1, 2, 3)), {"value": np.ones((1, 3))}, {}, TypeError, "mappings"),
        ({"value": np.ones((1, 2, 3))}, {"value": np.ones((1, 3))}, {"data_dim": 2}, ValueError, "out of range"),
        (
            {"value": np.ones((1, 2, 3))},
            {"value": np.ones((1, 3))},
            {"data_dim": "missing"},
            KeyError,
            "prediction key",
        ),
        ({"value": np.ones((1, 2, 3))}, {"other": np.ones((1, 3))}, {}, KeyError, "empirical key"),
        ({"value": np.ones((2, 3))}, {"value": np.ones((2, 3))}, {}, ValueError, "Predictive variable"),
        ({"value": np.ones((2, 2, 3))}, {"value": np.ones(3)}, {}, ValueError, "Empirical variable"),
        ({"value": np.ones((2, 2, 3))}, {"value": np.ones((3, 3))}, {}, ValueError, "must match"),
        (
            {"value": np.ones((1, 2, 3))},
            {"value": np.ones((1, 3))},
            {"dist_type": "violin"},
            ValueError,
            "dist_type",
        ),
        (
            {"value": np.ones((1, 2, 3))},
            {"value": np.ones((1, 3))},
            {"aggregation": np.mean, "aggregate_strategy": "invalid"},
            ValueError,
            "aggregate_strategy",
        ),
        (
            {"value": np.ones((1, 2, 3))},
            {"value": np.ones((1, 3))},
            {"num_cols": 0},
            ValueError,
            "num_cols",
        ),
    ],
    ids=[
        "not-mapping",
        "index",
        "prediction-name",
        "empirical-name",
        "prediction-shape",
        "empirical-shape",
        "shape-mismatch",
        "dist-type",
        "aggregate-strategy",
        "columns",
    ],
)
def test_plot_posterior_resimulation_validates_inputs(prediction, empirical, kwargs, error, message):
    with pytest.raises(error, match=message):
        plot_posterior_resimulation(prediction, empirical, **kwargs)


def test_posterior_resimulation_aggregate_spaghetti_uses_smoothed_dataset_centers():
    rng = np.random.default_rng(11)
    prediction = rng.normal(size=(3, 4, 7))
    empirical = rng.normal(size=(3, 7))

    fig = plot_posterior_resimulation(
        {"value": prediction},
        {"value": empirical},
        kind="time_series",
        aggregation=np.mean,
        smoothing="sma",
        smoothing_window=3,
        uncertainty_fun=None,
        marginal=False,
        spaghetti=True,
    )

    assert len(fig.axes[0].lines) == 5
    assert "Individual" in [text.get_text() for text in fig.legends[0].get_texts()]
    plt.close(fig)


def test_posterior_resimulation_individual_spaghetti_hides_unused_panel():
    rng = np.random.default_rng(12)
    prediction = rng.normal(size=(3, 2, 5))
    empirical = rng.normal(size=(3, 5))

    fig = plot_posterior_resimulation(
        {"value": prediction},
        {"value": empirical},
        kind="time_series",
        uncertainty_fun=None,
        marginal=False,
        spaghetti=True,
        num_cols=2,
    )

    assert [len(ax.lines) for ax in fig.axes[:3]] == [4, 4, 4]
    assert not fig.axes[3].axison
    plt.close(fig)


def test_posterior_resimulation_discrete_distribution_supports_no_epistemic_strategy():
    prediction = np.array(
        [
            np.zeros((3, 4)),
            np.ones((3, 4)),
        ]
    )
    empirical = np.array([[0, 0, 0, 0], [1, 1, 1, 1]])

    fig = plot_posterior_resimulation(
        {"choice": prediction},
        {"choice": empirical},
        kind="dist",
        aggregation=np.mean,
        aggregate_strategy="no_epistemic",
    )

    assert fig.axes[0].get_xticks().tolist() == [0, 1]
    assert sum(patch.get_height() for patch in fig.axes[0].patches) == pytest.approx(1.0)
    plt.close(fig)


def test_joint_prior_covers_mixture_shared_and_bounded_trajectory_panels():
    rng = np.random.default_rng(13)
    mixture_weights = rng.dirichlet([2.0, 1.0], size=50)

    fig = plot_joint_prior(
        local_params={"theta": rng.normal(size=(4, 8))},
        hyper_params={
            "theta_mixture_weights": mixture_weights,
            "theta_sigma": rng.normal(size=50),
        },
        shared_params={"offset": rng.normal(size=50)},
        param_bounds={"theta": (-2.0, 2.0)},
        mixture_names={"theta": ["smooth", "jump"]},
        marginal=True,
        num_bins=6,
    )

    trajectory_ax = next(ax for ax in fig.axes if ax.get_title() == "Trajectory")
    assert trajectory_ax.get_ylim() == pytest.approx((-2.0, 2.0))
    component_legend = next(ax.get_legend() for ax in fig.axes if ax.get_legend() is not None)
    assert [text.get_text() for text in component_legend.get_texts()] == ["smooth", "jump"]
    assert any(not ax.axison for ax in fig.axes)
    plt.close(fig)


def test_time_invariant_prior_labels_mixture_components_and_hides_unused_axis():
    weights = np.random.default_rng(14).dirichlet([1.0, 1.0], size=40)

    fig = plot_time_invariant_prior(
        hyper_params={"theta_mixture_weights": weights},
        shared_params={},
        mixture_names={"theta": ["random walk", "jump"]},
        num_cols=2,
        num_bins=5,
    )

    legend = fig.axes[0].get_legend()
    assert [text.get_text() for text in legend.get_texts()] == ["random walk", "jump"]
    assert not fig.axes[1].axison
    plt.close(fig)


@pytest.mark.parametrize(
    ("module", "wrapper_name", "bayesflow_name", "color_key"),
    [
        (recovery_module, "plot_recovery", "recovery", "color"),
        (calibration_module, "plot_calibration", "calibration_ecdf", "rank_ecdf_color"),
    ],
)
def test_time_invariant_wrappers_prepare_named_data_and_forward_options(
    monkeypatch,
    module,
    wrapper_name,
    bayesflow_name,
    color_key,
):
    captured = {}
    sentinel = object()

    def fake_plot(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(module.bf.diagnostics.plots, bayesflow_name, fake_plot)
    estimates = {"a": np.zeros((2, 3)), "b": np.ones((2, 3))}
    targets = {"a": np.zeros(2), "b": np.ones(2)}

    result = getattr(module, wrapper_name)(
        estimates,
        targets,
        variable_keys=["b", "a"],
        variable_names=["B", "A"],
        color="#123456",
        metric_fontsize=17,
        num_col=2,
    )

    assert result is sentinel
    assert captured["estimates"].shape == (2, 3, 2)
    assert captured["targets"].shape == (2, 2)
    assert captured["variable_names"] == ["B", "A"]
    assert captured[color_key] == "#123456"
    assert captured["metric_fontsize"] == 17
    assert captured["num_col"] == 2


def test_time_varying_verification_accepts_one_color_for_all_metrics():
    rng = np.random.default_rng(15)
    targets = rng.normal(size=(5, 3, 1))
    estimates = targets[:, None] + rng.normal(scale=0.1, size=(5, 8, 3, 1))

    fig = plot_time_varying_verification(estimates, targets, colors="#123456")

    assert all(ax.lines[0].get_color() == "#123456" for ax in fig.axes)
    plt.close(fig)


def test_time_varying_verification_rejects_wrong_number_of_colors():
    with pytest.raises(ValueError, match="4 entries"):
        plot_time_varying_verification(
            np.ones((2, 3, 4, 1)),
            np.ones((2, 4, 1)),
            colors=["red", "blue"],
        )
