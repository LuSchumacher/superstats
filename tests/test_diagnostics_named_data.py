import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

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
    )

    assert [ax.get_ylabel() for ax in fig.axes] == [expected_ylabel, "", expected_ylabel, ""]
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
        pred_data,
        real_data,
        data_dim="response_time",
        kind="trajectory",
        marginal=False,
        uncertainty_fun="std",
    )

    assert fig is not None
    plt.close(fig)
