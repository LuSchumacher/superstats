import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import superstats.diagnostics.plots.posterior_resimulation as posterior_resimulation_module
import superstats.diagnostics.plots.prior_push_forward as prior_push_forward_module
import superstats.diagnostics.plots.time_invariant_posterior as time_invariant_posterior_module
import superstats.diagnostics.plots.time_varying_posterior as time_varying_posterior_module
import superstats.diagnostics.plots.z_score_contraction as z_score_contraction_module
from superstats.defaults import (
    BASE_COL_WIDTH,
    BASE_ROW_HEIGHT,
    HSPACE,
    LABEL_FONTSIZE,
    TICK_FONTSIZE,
    TITLE_FONTSIZE,
    WSPACE,
)
from superstats.diagnostics.plots import (
    plot_posterior_resimulation,
    plot_push_forward,
    plot_time_invariant_posterior,
    plot_time_varying_prior,
    plot_time_varying_posterior,
    plot_z_score_contraction,
)


def test_plot_push_forward_accepts_named_data():
    rng = np.random.default_rng(0)
    data = {
        "response_time": rng.normal(size=(3, 5)),
        "choice": rng.integers(0, 2, size=(3, 5)).astype(float),
    }

    fig = plot_push_forward(
        data,
        data_dim="response_time",
        kind="time_series",
        marginal=False,
        uncertainty_fun=None,
    )

    assert fig is not None
    plt.close(fig)


def test_z_score_contraction_uses_shared_data_preparation_and_defaults(monkeypatch):
    captured = {}
    sentinel = object()

    def fake_z_score_contraction(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        z_score_contraction_module.bf.diagnostics.plots,
        "z_score_contraction",
        fake_z_score_contraction,
    )
    estimates = {
        "a": np.zeros((2, 4)),
        "b": np.ones((2, 4)),
    }
    targets = {
        "a": np.zeros(2),
        "b": np.ones(2),
    }

    result = plot_z_score_contraction(
        estimates=estimates,
        targets=targets,
        variable_keys=["b", "a"],
        variable_names=["B", "A"],
        markersize=7,
    )

    assert result is sentinel
    assert captured["estimates"].shape == (2, 4, 2)
    assert captured["targets"].shape == (2, 2)
    assert captured["variable_names"] == ["B", "A"]
    assert captured["color"] == "#356673"
    assert captured["title_fontsize"] == TITLE_FONTSIZE
    assert captured["label_fontsize"] == LABEL_FONTSIZE
    assert captured["tick_fontsize"] == TICK_FONTSIZE
    assert captured["markersize"] == 7


def test_time_varying_posterior_uses_shared_distribution_and_layout(monkeypatch):
    rng = np.random.default_rng(6)
    estimates = {"v": rng.normal(size=(2, 4, 6, 1))}
    targets = {"v": rng.normal(size=(2, 6, 1))}
    captured = []

    def fake_plot_dist(values, **kwargs):
        captured.append((np.asarray(values), kwargs))

    monkeypatch.setattr(
        time_varying_posterior_module,
        "plot_dist",
        fake_plot_dist,
    )

    fig = plot_time_varying_posterior(
        estimates=estimates,
        targets=targets,
        aggregation=np.mean,
        uncertainty_fun=None,
        dist_type="both",
        num_bins=7,
    )

    np.testing.assert_allclose(
        captured[0][0],
        estimates["v"][..., 0].reshape(-1),
    )
    np.testing.assert_allclose(
        captured[1][0],
        targets["v"][..., 0].mean(axis=0),
    )
    assert len(captured) == 2
    assert all(kwargs["dist_type"] == "both" for _, kwargs in captured)
    assert all(kwargs["num_bins"] == 7 for _, kwargs in captured)
    assert all(kwargs["alpha"] == 0.5 for _, kwargs in captured)
    assert all(kwargs["orientation"] == "vertical" for _, kwargs in captured)
    assert all(kwargs["hide_axis"] is True for _, kwargs in captured)
    np.testing.assert_allclose(
        fig.get_size_inches(),
        [BASE_COL_WIDTH, BASE_ROW_HEIGHT + 1.6],
    )

    legend_anchor = fig.legends[0].get_bbox_to_anchor().transformed(fig.transFigure.inverted())
    main_ax = next(ax for ax in fig.axes if ax.axison)
    assert legend_anchor.y0 * fig.get_size_inches()[1] == pytest.approx(0.1)
    assert fig.subplotpars.bottom * fig.get_size_inches()[1] == pytest.approx(1.6)
    assert fig.subplotpars.hspace == pytest.approx(HSPACE)
    assert fig.subplotpars.wspace == pytest.approx(WSPACE)
    assert main_ax.title.get_fontsize() == 22
    assert main_ax.xaxis.label.get_fontsize() == 18
    assert main_ax.yaxis.label.get_fontsize() == 18
    assert main_ax.get_ylabel() == "Parameter value"
    assert main_ax.get_xticklabels()[0].get_fontsize() == 16
    plt.close(fig)


def test_time_varying_posterior_single_marginal_is_opaque(monkeypatch):
    estimates = {"v": np.ones((2, 3, 4, 1))}
    captured_alphas = []

    def fake_plot_dist(values, **kwargs):
        captured_alphas.append(kwargs["alpha"])

    monkeypatch.setattr(
        time_varying_posterior_module,
        "plot_dist",
        fake_plot_dist,
    )

    fig = plot_time_varying_posterior(
        estimates=estimates,
        aggregation=np.mean,
        uncertainty_fun=None,
    )

    assert captured_alphas == [1.0]
    plt.close(fig)


def test_time_varying_posterior_hist_shows_target_marginal():
    rng = np.random.default_rng(8)
    estimates = {"v": rng.normal(size=(2, 4, 6, 1))}
    targets = {"v": rng.normal(size=(2, 6, 1))}

    fig = plot_time_varying_posterior(
        estimates=estimates,
        targets=targets,
        aggregation=np.median,
        uncertainty_fun=None,
        dist_type="hist",
        num_bins=7,
    )

    histogram_axes = [ax for ax in fig.axes if ax.patches]
    assert len(histogram_axes) == 2
    assert all(len(ax.patches) == 7 for ax in histogram_axes)
    assert all(patch.get_facecolor()[3] == pytest.approx(0.5) for ax in histogram_axes for patch in ax.patches)
    plt.close(fig)


def test_time_varying_posterior_selects_datasets_in_requested_order():
    estimates = {
        "v": np.broadcast_to(
            np.arange(3, dtype=float)[:, None, None, None],
            (3, 2, 4, 1),
        )
    }
    targets = {
        "v": np.broadcast_to(
            (np.arange(3, dtype=float) + 10)[:, None, None],
            (3, 4, 1),
        )
    }

    fig = plot_time_varying_posterior(
        estimates=estimates,
        targets=targets,
        data_idx=[2, 0],
        uncertainty_fun=None,
        marginal=False,
    )

    assert len(fig.axes) == 2
    assert [ax.get_title() for ax in fig.axes] == ["Dataset 2", "Dataset 0"]
    np.testing.assert_allclose(fig.axes[0].lines[0].get_ydata(), 2.0)
    np.testing.assert_allclose(fig.axes[0].lines[1].get_ydata(), 12.0)
    np.testing.assert_allclose(fig.axes[1].lines[0].get_ydata(), 0.0)
    np.testing.assert_allclose(fig.axes[1].lines[1].get_ydata(), 10.0)
    plt.close(fig)


def test_time_varying_posterior_explicit_num_cols_overrides_dataset_columns():
    estimates = {f"p{i}": np.full((2, 3, 4, 1), i, dtype=float) for i in range(3)}

    fig = plot_time_varying_posterior(
        estimates=estimates,
        data_idx=1,
        num_cols=3,
        marginal=False,
        uncertainty_fun=None,
    )

    assert len(fig.axes) == 3
    assert fig.get_size_inches()[0] == pytest.approx(BASE_COL_WIDTH * 3)
    assert [ax.get_title() for ax in fig.axes] == ["p0", "p1", "p2"]
    assert [ax.get_ylabel() for ax in fig.axes] == ["Parameter value", "", ""]
    plt.close(fig)


def test_time_invariant_posterior_uses_shared_distribution_and_layout(monkeypatch):
    rng = np.random.default_rng(9)
    estimates = {"a": rng.normal(size=(2, 4, 3, 1))}
    targets = {"a": rng.normal(size=(2, 1))}
    captured = []

    def fake_plot_dist(values, **kwargs):
        captured.append((np.asarray(values), kwargs))

    monkeypatch.setattr(
        time_invariant_posterior_module,
        "plot_dist",
        fake_plot_dist,
    )

    fig = plot_time_invariant_posterior(
        estimates=estimates,
        targets=targets,
        aggregation=np.mean,
        dist_type="both",
        num_bins=7,
        dist_alpha=0.35,
    )

    np.testing.assert_allclose(
        captured[0][0],
        estimates["a"].reshape(-1),
    )
    assert captured[0][1]["dist_type"] == "both"
    assert captured[0][1]["num_bins"] == 7
    assert captured[0][1]["alpha"] == 0.35
    np.testing.assert_allclose(
        fig.get_size_inches(),
        [BASE_COL_WIDTH, BASE_ROW_HEIGHT + 1.6],
    )

    ax = fig.axes[0]
    target_value = targets["a"].mean(axis=0).item()
    np.testing.assert_allclose(ax.lines[0].get_xdata(), target_value)
    assert ax.get_ylabel() == "Density"
    assert ax.get_xlabel() == "Value"
    assert ax.title.get_fontsize() == 22
    assert ax.xaxis.label.get_fontsize() == 18
    assert ax.yaxis.label.get_fontsize() == 18
    assert ax.get_xticklabels()[0].get_fontsize() == 16

    legend_anchor = fig.legends[0].get_bbox_to_anchor().transformed(fig.transFigure.inverted())
    assert [text.get_text() for text in fig.legends[0].get_texts()] == ["Target"]
    assert legend_anchor.y0 * fig.get_size_inches()[1] == pytest.approx(0.1)
    assert fig.subplotpars.bottom * fig.get_size_inches()[1] == pytest.approx(1.6)
    assert fig.subplotpars.hspace == pytest.approx(HSPACE)
    assert fig.subplotpars.wspace == pytest.approx(WSPACE)
    plt.close(fig)


def test_time_invariant_posterior_uses_overlay_alpha_for_mixtures(monkeypatch):
    estimates = {
        "a": np.ones((2, 3, 4, 1)),
        "w_mixture_weights": np.ones((2, 3, 4, 2)),
    }
    captured_alphas = []

    def fake_plot_dist(values, **kwargs):
        captured_alphas.append(kwargs["alpha"])

    monkeypatch.setattr(
        time_invariant_posterior_module,
        "plot_dist",
        fake_plot_dist,
    )

    fig = plot_time_invariant_posterior(
        estimates=estimates,
        aggregation=np.mean,
        mixture_names={"w": ["left", "right"]},
    )

    assert captured_alphas == [1.0, 0.5, 0.5]
    assert [text.get_text() for text in fig.legends[0].get_texts()] == ["left", "right"]
    plt.close(fig)


def test_time_invariant_posterior_selects_datasets_in_requested_order(monkeypatch):
    estimates = {
        "a": np.broadcast_to(
            np.arange(3, dtype=float)[:, None, None, None],
            (3, 2, 4, 1),
        )
    }
    targets = {
        "a": (np.arange(3, dtype=float) + 10)[:, None],
    }
    captured_values = []

    def fake_plot_dist(values, **kwargs):
        captured_values.append(np.asarray(values))

    monkeypatch.setattr(
        time_invariant_posterior_module,
        "plot_dist",
        fake_plot_dist,
    )

    fig = plot_time_invariant_posterior(
        estimates=estimates,
        targets=targets,
        data_idx=[2, 0],
    )

    assert [ax.get_title() for ax in fig.axes] == ["Dataset 2", "Dataset 0"]
    np.testing.assert_allclose(captured_values[0], 2.0)
    np.testing.assert_allclose(captured_values[1], 0.0)
    np.testing.assert_allclose(fig.axes[0].lines[0].get_xdata(), 12.0)
    np.testing.assert_allclose(fig.axes[1].lines[0].get_xdata(), 10.0)
    assert [ax.get_xlabel() for ax in fig.axes] == ["Value", "Value"]
    assert [ax.get_ylabel() for ax in fig.axes] == ["a", ""]
    plt.close(fig)


def test_time_invariant_posterior_explicit_num_cols_overrides_dataset_columns():
    estimates = {f"p{i}": np.full((2, 3, 4, 1), i, dtype=float) for i in range(3)}

    fig = plot_time_invariant_posterior(
        estimates=estimates,
        data_idx=1,
        num_cols=3,
    )

    assert len(fig.axes) == 3
    assert fig.get_size_inches()[0] == pytest.approx(BASE_COL_WIDTH * 3)
    assert [ax.get_title() for ax in fig.axes] == ["p0", "p1", "p2"]
    assert [ax.get_ylabel() for ax in fig.axes] == ["Density", "", ""]
    plt.close(fig)


def test_time_invariant_posterior_uses_dynamic_columns():
    rng = np.random.default_rng(10)
    estimates = {f"p{i}": rng.normal(size=(2, 3, 4, 1)) for i in range(6)}

    fig = plot_time_invariant_posterior(
        estimates=estimates,
        aggregation=np.mean,
        dist_type="hist",
    )

    assert len(fig.axes) == 6
    assert fig.get_size_inches()[0] == pytest.approx(BASE_COL_WIDTH * 3)
    assert [ax.get_ylabel() for ax in fig.axes] == ["Density", "", "", "Density", "", ""]
    plt.close(fig)


def test_time_invariant_posterior_long_single_column_labels_do_not_overlap():
    rng = np.random.default_rng(11)
    names = ["short", "a_much_longer_time_invariant_parameter_name"]
    estimates = {name: rng.normal(size=(1, 3, 4, 1)) for name in names}

    fig = plot_time_invariant_posterior(estimates=estimates)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    assert fig.get_size_inches()[0] > BASE_COL_WIDTH
    for ax, name in zip(fig.axes, names):
        label = ax.yaxis.label
        label_bbox = label.get_window_extent(renderer)
        panel_bbox = ax.get_window_extent(renderer)

        assert label.get_text() == name
        assert label.get_horizontalalignment() == "right"
        assert label_bbox.x0 >= 0
        assert label_bbox.x1 < panel_bbox.x0

    plt.close(fig)


def test_plot_posterior_resimulation_selects_datasets_in_requested_order():
    pred_values = np.broadcast_to(
        np.arange(3, dtype=float)[:, None, None],
        (3, 2, 4),
    )
    real_values = np.broadcast_to(
        (np.arange(3, dtype=float) + 10)[:, None],
        (3, 4),
    )

    fig = plot_posterior_resimulation(
        {"value": pred_values},
        {"value": real_values},
        data_idx=[2, 0],
        kind="time_series",
        marginal=False,
        uncertainty_fun=None,
        num_cols=2,
    )

    assert [ax.get_title() for ax in fig.axes] == ["Dataset 2", "Dataset 0"]
    np.testing.assert_allclose(fig.axes[0].lines[0].get_ydata(), 2.0)
    np.testing.assert_allclose(fig.axes[0].lines[1].get_ydata(), 12.0)
    np.testing.assert_allclose(fig.axes[1].lines[0].get_ydata(), 0.0)
    np.testing.assert_allclose(fig.axes[1].lines[1].get_ydata(), 10.0)
    plt.close(fig)


def test_plot_posterior_resimulation_caps_columns_to_selected_datasets():
    pred_values = np.ones((3, 2, 4))
    real_values = np.ones((3, 4))

    fig = plot_posterior_resimulation(
        {"value": pred_values},
        {"value": real_values},
        data_idx=1,
        kind="time_series",
        smoothing="sma",
        marginal=False,
        uncertainty_fun=None,
    )

    assert len(fig.axes) == 1
    assert fig.get_size_inches()[0] == pytest.approx(BASE_COL_WIDTH)
    plt.close(fig)


def test_plot_posterior_resimulation_respects_explicit_num_cols():
    pred_values = np.ones((3, 2, 4))
    real_values = np.ones((3, 4))

    fig = plot_posterior_resimulation(
        {"value": pred_values},
        {"value": real_values},
        data_idx=1,
        kind="dist",
        num_cols=3,
    )

    assert len(fig.axes) == 3
    assert fig.get_size_inches()[0] == pytest.approx(BASE_COL_WIDTH * 3)
    assert [ax.get_title() for ax in fig.axes] == ["Dataset 1", "", ""]
    plt.close(fig)


def test_aggregate_time_series_plots_use_same_figure_dimensions():
    rng = np.random.default_rng(5)
    empirical = rng.normal(size=(3, 8))
    prediction = rng.normal(size=(3, 4, 8))

    push_forward_fig = plot_push_forward(
        {"value": empirical},
        kind="time_series",
        dist_type="kde",
        aggregation=np.mean,
        uncertainty_fun=None,
    )
    resimulation_fig = plot_posterior_resimulation(
        prediction={"value": prediction},
        empirical={"value": empirical},
        aggregation=np.mean,
        smoothing="ema",
        aggregate_strategy="full_uncertainty",
        dist_type="kde",
    )
    posterior_fig = plot_time_varying_posterior(
        estimates={"value": prediction[..., None]},
        targets={"value": empirical[..., None]},
        aggregation=np.mean,
        uncertainty_fun=None,
        dist_type="kde",
    )
    prior_fig = plot_time_varying_prior(
        {"value": prediction.reshape(-1, prediction.shape[-1])},
        dist_type="kde",
    )

    figures = [
        push_forward_fig,
        resimulation_fig,
        posterior_fig,
        prior_fig,
    ]
    for fig in figures:
        np.testing.assert_allclose(
            fig.get_size_inches(),
            [BASE_COL_WIDTH, BASE_ROW_HEIGHT + 1.6],
        )
        ax = next(ax for ax in fig.axes if ax.axison)
        assert ax.xaxis.label.get_fontsize() == LABEL_FONTSIZE
        assert ax.yaxis.label.get_fontsize() == LABEL_FONTSIZE
        assert ax.get_xticklabels()[0].get_fontsize() == TICK_FONTSIZE
        assert fig.legends[0].get_texts()[0].get_fontsize() == LABEL_FONTSIZE

    posterior_ax = next(ax for ax in posterior_fig.axes if ax.axison)
    prior_ax = next(ax for ax in prior_fig.axes if ax.axison)
    assert posterior_ax.title.get_fontsize() == TITLE_FONTSIZE
    assert prior_ax.title.get_fontsize() == TITLE_FONTSIZE

    for fig in figures:
        plt.close(fig)


def test_plot_posterior_resimulation_omits_legend_for_zero_width_hdi():
    trajectory = np.linspace(0.1, 1.1, 5)
    pred_values = np.broadcast_to(trajectory, (3, 1, 5))
    real_values = np.broadcast_to(trajectory, (3, 5))

    fig = plot_posterior_resimulation(
        {"value": pred_values},
        {"value": real_values},
        data_idx=1,
        kind="time_series",
        smoothing="sma",
        aggregate_strategy="no_epistemic",
        dist_type="kde",
    )

    labels = [text.get_text() for text in fig.legends[0].get_texts()]
    assert "95% HDI" not in labels
    plt.close(fig)


@pytest.mark.parametrize(
    ("kind", "batch_size", "expected_cols"),
    [
        ("time_series", 3, 3),
        ("dist", 6, 3),
        ("dist", 9, 3),
        ("dist", 11, 4),
    ],
)
def test_plot_push_forward_selects_dynamic_columns(
    kind,
    batch_size,
    expected_cols,
):
    data = {"value": np.random.default_rng(0).normal(size=(batch_size, 5))}

    fig = plot_push_forward(
        data,
        kind=kind,
        marginal=False,
        uncertainty_fun=None,
        dist_type="hist",
    )

    expected_rows = int(np.ceil(batch_size / expected_cols))
    assert len(fig.axes) == expected_rows * expected_cols
    assert fig.get_size_inches()[0] == pytest.approx(BASE_COL_WIDTH * expected_cols)
    plt.close(fig)


@pytest.mark.parametrize("dist_type", ["hist", "kde", "both"])
def test_plot_push_forward_supports_dist_type(dist_type):
    data = {"value": np.random.default_rng(0).normal(size=(20, 10))}

    fig = plot_push_forward(
        data,
        kind="dist",
        aggregation=np.mean,
        uncertainty_fun=None,
        dist_type=dist_type,
        num_bins=7,
    )

    ax = fig.axes[0]
    if dist_type in {"hist", "both"}:
        assert ax.patches
        assert len(ax.patches) == 7
    if dist_type == "hist":
        assert sum(patch.get_width() * patch.get_height() for patch in ax.patches) == pytest.approx(1.0)
    if dist_type == "kde":
        assert ax.collections
    if dist_type == "both":
        assert ax.lines
    assert ax.get_ylabel() == "Density"
    assert ax.get_xlabel() == "Parameter value"
    plt.close(fig)


def test_plot_push_forward_dist_labels_ticks_on_every_row_without_shared_x_axis():
    data = {
        "value": np.array(
            [
                np.linspace(0.0, 1.0, 20),
                np.linspace(10.0, 12.0, 20),
                np.linspace(100.0, 105.0, 20),
                np.linspace(1000.0, 1010.0, 20),
            ]
        )
    }

    fig = plot_push_forward(
        data,
        kind="dist",
        num_cols=2,
        dist_type="hist",
        num_bins=5,
        uncertainty_fun=None,
    )
    fig.canvas.draw()

    assert not fig.axes[0].get_shared_x_axes().joined(fig.axes[0], fig.axes[2])
    assert fig.axes[0].get_xlim() != pytest.approx(fig.axes[2].get_xlim())
    assert all(any(tick.get_visible() for tick in ax.get_xticklabels()) for ax in fig.axes)
    assert [ax.get_title() for ax in fig.axes] == [
        "Dataset 0",
        "Dataset 1",
        "Dataset 2",
        "Dataset 3",
    ]
    assert [ax.get_xlabel() for ax in fig.axes] == ["", "", "Parameter value", "Parameter value"]
    plt.close(fig)


@pytest.mark.parametrize(
    ("kind", "dist_type", "expected_ylabel"),
    [
        ("time_series", "hist", "Value"),
        ("dist", "hist", "Density"),
        ("dist", "kde", "Density"),
    ],
)
def test_plot_push_forward_labels_only_first_column(
    kind,
    dist_type,
    expected_ylabel,
):
    data = {"value": np.random.default_rng(0).normal(size=(4, 5))}

    fig = plot_push_forward(
        data,
        kind=kind,
        marginal=False,
        uncertainty_fun=None,
        num_cols=2,
        dist_type=dist_type,
    )

    assert [ax.get_ylabel() for ax in fig.axes] == [expected_ylabel, "", expected_ylabel, ""]
    assert fig.subplotpars.hspace == pytest.approx(HSPACE)
    assert fig.subplotpars.wspace == pytest.approx(WSPACE)
    plt.close(fig)


def test_plot_push_forward_discrete_histogram_uses_density():
    data = {"value": np.array([[0, 0, 1, 1, 1]])}

    fig = plot_push_forward(
        data,
        kind="dist",
        marginal=False,
        uncertainty_fun=None,
        dist_type="hist",
    )

    ax = fig.axes[0]
    assert sum(patch.get_height() for patch in ax.patches) == pytest.approx(1.0)
    assert ax.get_ylabel() == "Density"
    plt.close(fig)


def test_plot_push_forward_positions_aggregate_legend():
    data = {"value": np.random.default_rng(0).normal(size=(10, 5))}

    fig = plot_push_forward(
        data,
        kind="time_series",
        aggregation=np.mean,
        uncertainty_fun="std",
        marginal=False,
    )

    legend_anchor = fig.legends[0].get_bbox_to_anchor().transformed(fig.transFigure.inverted())
    figure_height = fig.get_size_inches()[1]

    assert fig.axes[0].get_ylabel() == "Value"
    assert legend_anchor.y0 * figure_height == pytest.approx(0.1)
    assert fig.subplotpars.bottom * figure_height == pytest.approx(1.6)
    plt.close(fig)


def test_aggregate_time_series_marginal_uses_center_per_step(
    monkeypatch,
):
    values = np.arange(24, dtype=float).reshape(4, 6) / 10
    captured = {}

    def fake_plot_dist(plotted_values, **kwargs):
        captured["values"] = np.asarray(plotted_values)

    monkeypatch.setattr(
        prior_push_forward_module,
        "plot_dist",
        fake_plot_dist,
    )

    fig = plot_push_forward(
        {"value": values},
        kind="time_series",
        aggregation=np.mean,
        uncertainty_fun=None,
        marginal=True,
        dist_type="kde",
    )

    np.testing.assert_allclose(
        captured["values"],
        values.mean(axis=0),
    )
    plt.close(fig)


def test_plot_push_forward_rejects_trajectory_kind():
    with pytest.raises(ValueError, match="time_series"):
        plot_push_forward(
            {"value": np.ones((2, 3))},
            kind="trajectory",
        )


def test_plot_posterior_resimulation_rejects_trajectory_kind():
    with pytest.raises(ValueError, match="time_series"):
        plot_posterior_resimulation(
            prediction={"value": np.ones((1, 2, 3))},
            empirical={"value": np.ones((1, 3))},
            kind="trajectory",
        )


def test_plot_posterior_resimulation_accepts_named_data():
    rng = np.random.default_rng(1)
    pred_data = {
        "response_time": rng.normal(size=(2, 4, 5)),
        "choice": rng.integers(0, 2, size=(2, 4, 5)).astype(float),
    }
    real_data = {
        "response_time": rng.normal(size=(2, 5)),
        "choice": rng.integers(0, 2, size=(2, 5)).astype(float),
    }

    fig = plot_posterior_resimulation(
        prediction=pred_data,
        empirical=real_data,
        data_dim="response_time",
        kind="time_series",
        marginal=False,
        uncertainty_fun="std",
    )

    assert fig is not None
    plt.close(fig)


@pytest.mark.parametrize(
    "aggregate_strategy",
    ["full_uncertainty", "no_epistemic"],
)
def test_aggregate_resimulation_marginal_uses_ribbon_pool(
    monkeypatch,
    aggregate_strategy,
):
    rng = np.random.default_rng(2)
    pred_values = rng.normal(size=(2, 3, 5))
    real_values = rng.normal(size=(2, 5))
    captured = {
        "values": [],
        "alphas": [],
    }

    def fake_plot_dist(plotted_values, **kwargs):
        captured["values"].append(np.asarray(plotted_values))
        captured["alphas"].append(kwargs["alpha"])

    monkeypatch.setattr(
        posterior_resimulation_module,
        "plot_dist",
        fake_plot_dist,
    )

    fig = plot_posterior_resimulation(
        {"value": pred_values},
        {"value": real_values},
        kind="time_series",
        aggregation=np.mean,
        aggregate_strategy=aggregate_strategy,
        uncertainty_fun=None,
        marginal=True,
        dist_alpha=0.35,
        dist_type="kde",
    )

    expected_values = (
        pred_values.reshape(-1) if aggregate_strategy == "full_uncertainty" else pred_values.mean(axis=1).reshape(-1)
    )
    np.testing.assert_allclose(captured["values"][0], expected_values)
    np.testing.assert_allclose(
        captured["values"][1],
        real_values.mean(axis=0),
    )
    assert captured["alphas"] == [0.35, 0.35]
    plt.close(fig)


def test_aggregate_resimulation_hist_shows_empirical_distribution():
    rng = np.random.default_rng(7)
    prediction = rng.normal(size=(3, 50, 8))
    empirical = rng.normal(size=(3, 8))

    fig = plot_posterior_resimulation(
        prediction={"value": prediction},
        empirical={"value": empirical},
        aggregation=np.mean,
        uncertainty_fun=None,
        dist_type="hist",
        num_bins=7,
    )

    histogram_axes = [ax for ax in fig.axes if ax.patches]
    assert len(histogram_axes) == 2
    assert all(len(ax.patches) == 7 for ax in histogram_axes)
    labels = [text.get_text() for text in fig.legends[0].get_texts()]
    assert "Empirical" in labels
    assert "Real data" not in labels
    plt.close(fig)


def test_posterior_resimulation_uses_shared_legend_and_spacing():
    rng = np.random.default_rng(3)
    pred_values = rng.normal(size=(2, 3, 5))
    real_values = rng.normal(size=(2, 5))

    fig = plot_posterior_resimulation(
        {"value": pred_values},
        {"value": real_values},
        kind="time_series",
        aggregation=np.mean,
        uncertainty_fun=None,
        marginal=False,
    )

    legend_anchor = fig.legends[0].get_bbox_to_anchor().transformed(fig.transFigure.inverted())
    figure_height = fig.get_size_inches()[1]

    assert fig.axes[0].get_ylabel() == "Value"
    assert legend_anchor.y0 * figure_height == pytest.approx(0.1)
    assert fig.subplotpars.bottom * figure_height == pytest.approx(1.6)
    assert fig.subplotpars.hspace == pytest.approx(HSPACE)
    assert fig.subplotpars.wspace == pytest.approx(WSPACE)
    plt.close(fig)


def test_posterior_resimulation_dist_uses_num_bins_and_density_label():
    rng = np.random.default_rng(4)
    pred_values = rng.normal(size=(2, 10, 5))
    real_values = rng.normal(size=(2, 5))

    fig = plot_posterior_resimulation(
        {"value": pred_values},
        {"value": real_values},
        kind="dist",
        aggregation=np.mean,
        dist_type="hist",
        num_bins=7,
    )

    ax = fig.axes[0]
    assert len(ax.patches) == 7
    assert ax.get_ylabel() == "Density"
    plt.close(fig)


def test_posterior_resimulation_dist_uses_independent_x_axes_and_dataset_titles():
    prediction = {
        "value": np.array(
            [
                np.tile(np.linspace(0.0, 1.0, 20), (2, 1)),
                np.tile(np.linspace(10.0, 12.0, 20), (2, 1)),
                np.tile(np.linspace(100.0, 105.0, 20), (2, 1)),
                np.tile(np.linspace(1000.0, 1010.0, 20), (2, 1)),
            ]
        )
    }
    empirical = {
        "value": np.array(
            [
                np.linspace(0.0, 1.0, 20),
                np.linspace(10.0, 12.0, 20),
                np.linspace(100.0, 105.0, 20),
                np.linspace(1000.0, 1010.0, 20),
            ]
        )
    }

    fig = plot_posterior_resimulation(
        prediction=prediction,
        empirical=empirical,
        kind="dist",
        data_idx=[3, 1, 0, 2],
        num_cols=2,
        dist_type="hist",
        num_bins=5,
        title_fontsize=19,
    )
    fig.canvas.draw()

    assert [ax.get_title() for ax in fig.axes] == [
        "Dataset 3",
        "Dataset 1",
        "Dataset 0",
        "Dataset 2",
    ]
    assert all(ax.title.get_fontsize() == 19 for ax in fig.axes)
    assert not fig.axes[0].get_shared_x_axes().joined(fig.axes[0], fig.axes[2])
    assert fig.axes[0].get_xlim() != pytest.approx(fig.axes[2].get_xlim())
    assert all(any(tick.get_visible() for tick in ax.get_xticklabels()) for ax in fig.axes)
    plt.close(fig)


def test_posterior_resimulation_aggregate_strategies_pool_as_documented():
    pred_values = np.array(
        [
            [[0.0, 0.0], [0.0, 0.0], [100.0, 100.0]],
            [[10.0, 10.0], [10.0, 10.0], [10.0, 10.0]],
        ]
    )
    real_values = np.zeros((2, 2))

    full_fig = plot_posterior_resimulation(
        {"value": pred_values},
        {"value": real_values},
        kind="time_series",
        aggregation=np.median,
        aggregate_strategy="full_uncertainty",
        uncertainty_fun=None,
        marginal=False,
    )
    collapsed_fig = plot_posterior_resimulation(
        {"value": pred_values},
        {"value": real_values},
        kind="time_series",
        aggregation=np.median,
        aggregate_strategy="no_epistemic",
        uncertainty_fun=None,
        marginal=False,
    )

    np.testing.assert_allclose(
        full_fig.axes[0].lines[0].get_ydata(),
        [10.0, 10.0],
    )
    np.testing.assert_allclose(
        collapsed_fig.axes[0].lines[0].get_ydata(),
        [5.0, 5.0],
    )
    plt.close(full_fig)
    plt.close(collapsed_fig)
