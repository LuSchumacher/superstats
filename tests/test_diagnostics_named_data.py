import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import superstats.diagnostics.plots.posterior_resimulation as posterior_resimulation_module
import superstats.diagnostics.plots.prior_push_forward as prior_push_forward_module
from superstats.defaults import BASE_COL_WIDTH
from superstats.diagnostics.plots import plot_posterior_resimulation, plot_push_forward


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


def test_aggregate_time_series_plots_use_same_figure_dimensions():
    rng = np.random.default_rng(5)
    empiric = rng.normal(size=(3, 8))
    prediction = rng.normal(size=(3, 4, 8))

    push_forward_fig = plot_push_forward(
        {"value": empiric},
        kind="time_series",
        dist_type="kde",
        aggregation=np.mean,
        uncertainty_fun=None,
    )
    resimulation_fig = plot_posterior_resimulation(
        prediction={"value": prediction},
        empiric={"value": empiric},
        aggregation=np.mean,
        smoothing="ema",
        aggregate_strategy="full_uncertainty",
        dist_type="kde",
    )

    np.testing.assert_allclose(
        resimulation_fig.get_size_inches(),
        push_forward_fig.get_size_inches(),
    )
    push_forward_ax = next(ax for ax in push_forward_fig.axes if ax.axison)
    resimulation_ax = next(ax for ax in resimulation_fig.axes if ax.axison)
    assert resimulation_ax.xaxis.label.get_fontsize() == push_forward_ax.xaxis.label.get_fontsize()
    assert resimulation_ax.get_xticklabels()[0].get_fontsize() == push_forward_ax.get_xticklabels()[0].get_fontsize()
    assert (
        resimulation_fig.legends[0].get_texts()[0].get_fontsize()
        == push_forward_fig.legends[0].get_texts()[0].get_fontsize()
    )
    plt.close(push_forward_fig)
    plt.close(resimulation_fig)


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
        assert sum(patch.get_height() for patch in ax.patches) == pytest.approx(20)
    if dist_type == "kde":
        assert ax.collections
    if dist_type == "both":
        assert ax.lines
    assert ax.get_ylabel() == ("Count" if dist_type == "hist" else "Density")
    plt.close(fig)


@pytest.mark.parametrize(
    ("kind", "dist_type", "expected_ylabel"),
    [
        ("time_series", "hist", "Value"),
        ("dist", "hist", "Count"),
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
        hspace=0.6,
        wspace=0.1,
    )

    assert [ax.get_ylabel() for ax in fig.axes] == [expected_ylabel, "", expected_ylabel, ""]
    assert fig.subplotpars.hspace == pytest.approx(0.6)
    assert fig.subplotpars.wspace == pytest.approx(0.1)
    plt.close(fig)


def test_plot_push_forward_discrete_histogram_uses_counts():
    data = {"value": np.array([[0, 0, 1, 1, 1]])}

    fig = plot_push_forward(
        data,
        kind="dist",
        marginal=False,
        uncertainty_fun=None,
        dist_type="hist",
    )

    ax = fig.axes[0]
    assert sum(patch.get_height() for patch in ax.patches) == pytest.approx(5)
    assert ax.get_ylabel() == "Count"
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
            empiric={"value": np.ones((1, 3))},
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
        empiric=real_data,
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
        hspace=0.6,
        wspace=0.1,
    )

    legend_anchor = fig.legends[0].get_bbox_to_anchor().transformed(fig.transFigure.inverted())
    figure_height = fig.get_size_inches()[1]

    assert fig.axes[0].get_ylabel() == "Value"
    assert legend_anchor.y0 * figure_height == pytest.approx(0.1)
    assert fig.subplotpars.bottom * figure_height == pytest.approx(1.6)
    assert fig.subplotpars.hspace == pytest.approx(0.6)
    assert fig.subplotpars.wspace == pytest.approx(0.1)
    plt.close(fig)


def test_posterior_resimulation_dist_uses_num_bins_and_count_label():
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
    assert ax.get_ylabel() == "Count"
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
