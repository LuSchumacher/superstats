import numpy as np
import pytest
import matplotlib.pyplot as plt
from matplotlib import font_manager

import superstats.utils.plotting as plotting_module
from superstats.defaults import (
    BASE_COLOR,
    BASE_COL_WIDTH,
    BASE_ROW_HEIGHT,
    CATEGORICAL_PALETTE,
    DIST_ALPHA,
    HSPACE,
    JOINT_HSPACE,
    LABEL_FONTSIZE,
    LABEL_PAD,
    METRIC_COLORS,
    OVERLAY_DIST_ALPHA,
    TICK_FONTSIZE,
    TITLE_FONTSIZE,
    WSPACE,
    Y_LABEL_PAD,
)
from superstats.diagnostics.metrics import (
    calibration_error_per_step,
    correlation_per_step,
    nrmse_per_step,
    posterior_contraction_per_step,
)
from superstats.diagnostics.plots.joint_prior import plot_joint_prior
from superstats.diagnostics.plots.time_invariant_prior import plot_time_invariant_prior
from superstats.diagnostics.plots.time_varying_prior import plot_time_varying_prior
from superstats.diagnostics.plots.time_varying_verification import (
    plot_time_varying_verification,
)
from superstats.networks.utils import expand_singletons_to_common_length
from superstats.utils.plotting import (
    compute_uncertainty_band,
    compute_uncertainty_bands,
    get_default_num_cols,
    get_layout,
    plot_dist,
    plot_uncertainty_band,
    plot_uncertainty_bands,
    prepare_plot_data,
    smooth_trajectories,
)


def test_expand_singletons_broadcasts_scalars_and_singletons():
    result = expand_singletons_to_common_length(width=16, activation=["relu", "tanh"], bias=[True])

    assert result == {"width": [16, 16], "activation": ["relu", "tanh"], "bias": [True, True]}


@pytest.mark.parametrize("kwargs", [{"a": []}, {"a": [1, 2], "b": [3, 4, 5]}])
def test_expand_singletons_rejects_empty_or_incompatible_sequences(kwargs):
    with pytest.raises(ValueError):
        expand_singletons_to_common_length(**kwargs)


def test_prepare_plot_data_selects_and_stacks_named_variables():
    estimates = {"b": np.ones((2, 3, 4)), "a": np.zeros((2, 3, 4))}
    targets = {"b": np.ones((2, 4)), "a": np.zeros((2, 4))}

    est, target, names = prepare_plot_data(estimates, targets, variable_keys=["a", "b"], variable_names=["A", "B"])

    assert names == ["A", "B"]
    assert est.shape == (2, 3, 4, 2)
    assert target.shape == (2, 4, 2)
    np.testing.assert_array_equal(est[..., 0], 0)
    np.testing.assert_array_equal(target[..., 1], 1)


def test_prepare_plot_data_rejects_mixed_inputs_and_unknown_names():
    with pytest.raises(ValueError, match="must both"):
        prepare_plot_data({"x": np.ones(2)}, np.ones((2, 1)))
    with pytest.raises(ValueError, match="not found"):
        prepare_plot_data({"x": np.ones(2)}, {"x": np.ones(2)}, variable_keys=["missing"])


@pytest.mark.parametrize("num_rows", [1, 2, 3])
def test_layout_preserves_physical_legend_spacing(num_rows):
    figsize, legend_bottom, legend_y = get_layout(
        num_rows,
        num_cols=2,
        figsize=None,
        col_width=BASE_COL_WIDTH,
        row_height=BASE_ROW_HEIGHT,
    )

    assert figsize[0] == pytest.approx(BASE_COL_WIDTH * 2)
    assert (figsize[1] - 1.6) / num_rows == pytest.approx(BASE_ROW_HEIGHT)
    assert legend_bottom * figsize[1] == pytest.approx(1.6)
    assert legend_y * figsize[1] == pytest.approx(0.1)


def test_layout_preserves_explicit_figure_size():
    figsize, legend_bottom, legend_y = get_layout(
        num_rows=2,
        num_cols=3,
        figsize=(9.0, 7.0),
        col_width=BASE_COL_WIDTH,
        row_height=BASE_ROW_HEIGHT,
    )

    assert figsize == (9.0, 7.0)
    assert legend_bottom * figsize[1] == pytest.approx(1.6)
    assert legend_y * figsize[1] == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("num_panels", "expected_cols"),
    [
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 2),
        (5, 3),
        (6, 3),
        (7, 4),
        (8, 4),
        (9, 3),
        (10, 4),
        (11, 4),
    ],
)
def test_default_columns(num_panels, expected_cols):
    assert get_default_num_cols(num_panels) == expected_cols


def test_shared_trajectory_smoothing_supports_sma_and_ema():
    values = np.array([[1.0, 2.0, 3.0, 4.0]])

    np.testing.assert_allclose(
        smooth_trajectories(values, "sma", smoothing_window=2),
        [[1.0, 1.5, 2.5, 3.5]],
    )
    np.testing.assert_allclose(
        smooth_trajectories(values, "ema", smoothing_window=3),
        [[1.0, 1.5, 2.25, 3.125]],
    )
    assert smooth_trajectories(values, None) is values


def test_shared_uncertainty_band_computes_and_draws_visible_intervals():
    trajectories = np.array(
        [
            [0.0, 1.0],
            [2.0, 3.0],
        ]
    )
    center = trajectories.mean(axis=0)
    lower, upper = compute_uncertainty_band(
        trajectories,
        "std",
        center,
    )
    fig, ax = plt.subplots()

    visible = plot_uncertainty_band(
        ax,
        np.arange(2),
        lower,
        upper,
        BASE_COLOR,
        alpha=0.3,
    )

    assert visible is True
    assert ax.collections
    plt.close(fig)


@pytest.mark.parametrize(
    ("uncertainty_fun", "expected_outer", "expected_inner"),
    [
        ("std", 1.0, 0.5),
        ("mad", 1.48, 0.74),
    ],
)
def test_shared_uncertainty_bands_have_nested_normal_scale_intervals(
    uncertainty_fun,
    expected_outer,
    expected_inner,
):
    trajectories = np.array([[-1.0], [0.0], [1.0]])
    center = np.array([0.0])

    outer, inner = compute_uncertainty_bands(trajectories, uncertainty_fun, center)

    raw_scale = trajectories.std(axis=0) if uncertainty_fun == "std" else np.array([1.0])
    np.testing.assert_allclose(outer[0], -expected_outer * raw_scale)
    np.testing.assert_allclose(outer[1], expected_outer * raw_scale)
    np.testing.assert_allclose(inner[0], -expected_inner * raw_scale)
    np.testing.assert_allclose(inner[1], expected_inner * raw_scale)


@pytest.mark.parametrize("uncertainty_fun", ["ci", "hdi"])
def test_shared_interval_uncertainty_bands_use_95_and_65_percent(uncertainty_fun):
    trajectories = np.arange(100, dtype=float)[:, None]
    center = np.array([49.5])

    outer, inner = compute_uncertainty_bands(trajectories, uncertainty_fun, center)

    assert outer[0][0] <= inner[0][0] < inner[1][0] <= outer[1][0]
    assert inner[1][0] - inner[0][0] < outer[1][0] - outer[0][0]


def test_shared_nested_uncertainty_renderer_draws_two_ribbons():
    fig, ax = plt.subplots()

    visible = plot_uncertainty_bands(
        ax,
        np.arange(2),
        (np.array([-1.0, -1.0]), np.array([1.0, 1.0])),
        (np.array([-0.5, -0.5]), np.array([0.5, 0.5])),
        BASE_COLOR,
        alpha=0.4,
    )

    assert visible is True
    assert len(ax.collections) == 2
    assert ax.collections[0].get_alpha() == pytest.approx(0.2)
    assert ax.collections[1].get_alpha() == pytest.approx(0.4)
    plt.close(fig)


@pytest.mark.parametrize("dist_type", ["hist", "kde", "both"])
@pytest.mark.parametrize("orientation", ["horizontal", "vertical"])
def test_plot_dist_supports_all_types_and_orientations(dist_type, orientation):
    fig, ax = plt.subplots()
    values = np.random.default_rng(0).normal(size=100)

    plot_dist(values, ax=ax, dist_type=dist_type, color=BASE_COLOR, orientation=orientation)

    if dist_type in {"hist", "both"}:
        assert ax.patches
    if dist_type == "hist":
        density_integral = sum(patch.get_width() * patch.get_height() for patch in ax.patches)
        assert density_integral == pytest.approx(1.0)
    if dist_type == "kde":
        assert ax.collections
        assert all(np.allclose(collection.get_linewidths(), 0) for collection in ax.collections)
    if dist_type == "both":
        assert ax.lines
    plt.close(fig)


def test_plot_dist_rejects_invalid_options():
    fig, ax = plt.subplots()

    with pytest.raises(ValueError, match="dist_type"):
        plot_dist(np.ones(3), ax=ax, dist_type="invalid", color=BASE_COLOR)
    with pytest.raises(ValueError, match="orientation"):
        plot_dist(np.ones(3), ax=ax, dist_type="hist", color=BASE_COLOR, orientation="invalid")
    with pytest.raises(ValueError, match="num_bins"):
        plot_dist(np.ones(3), ax=ax, dist_type="hist", color=BASE_COLOR, num_bins=0)
    with pytest.raises(ValueError, match="alpha"):
        plot_dist(np.ones(3), ax=ax, dist_type="hist", color=BASE_COLOR, alpha=1.1)
    plt.close(fig)


def test_plot_dist_omits_bins_by_default(monkeypatch):
    captured = {}

    def fake_histplot(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(plotting_module.sns, "histplot", fake_histplot)

    fig, ax = plt.subplots()
    plot_dist(
        np.arange(10),
        ax=ax,
        dist_type="hist",
        color=BASE_COLOR,
    )

    assert "bins" not in captured
    assert captured["stat"] == "density"
    plt.close(fig)


@pytest.mark.parametrize("dist_type", ["hist", "kde", "both"])
def test_time_invariant_prior_supports_dist_type(dist_type):
    fig = plot_time_invariant_prior(
        hyper_params={"mu": np.random.default_rng(0).normal(size=100)},
        shared_params={},
        dist_type=dist_type,
        num_bins=7,
        num_cols=1,
    )

    ax = fig.axes[0]
    assert ax.get_xlabel() == "Value"
    assert ax.get_ylabel() == "Density"
    if dist_type in {"hist", "both"}:
        assert len(ax.patches) == 7
    plt.close(fig)


@pytest.mark.parametrize(
    ("num_params", "expected_cols"),
    [
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 2),
        (5, 3),
        (6, 3),
        (7, 4),
        (8, 4),
        (9, 3),
        (10, 4),
        (11, 4),
    ],
)
def test_time_invariant_prior_selects_compact_default_columns(num_params, expected_cols):
    rng = np.random.default_rng(0)
    hyper_params = {f"p{i}": rng.normal(size=30) for i in range(num_params)}

    fig = plot_time_invariant_prior(hyper_params, {}, dist_type="hist")

    expected_rows = int(np.ceil(num_params / expected_cols))
    assert len(fig.axes) == expected_rows * expected_cols
    assert fig.get_size_inches()[0] == pytest.approx(BASE_COL_WIDTH * expected_cols)
    plt.close(fig)


def test_time_varying_prior_labels_y_axis():
    fig = plot_time_varying_prior(
        {"v": np.random.default_rng(0).normal(size=(5, 10))},
        marginal=False,
    )

    assert fig.axes[0].get_ylabel() == "Parameter value"
    assert fig.axes[0].yaxis.labelpad == Y_LABEL_PAD
    plt.close(fig)


def test_time_varying_verification_labels_only_first_column():
    rng = np.random.default_rng(0)
    targets = rng.normal(size=(6, 5, 2))
    estimates = targets[:, None] + rng.normal(scale=0.2, size=(6, 20, 5, 2))

    fig = plot_time_varying_verification(
        estimates,
        targets,
        variable_names=["A", "B"],
    )

    expected_labels = [
        "Correlation\n(Truth vs. Estimate)",
        "NRMSE",
        "Posterior\nContraction",
        "Calibration\nError",
    ]
    for row_i, expected_label in enumerate(expected_labels):
        first_col = fig.axes[row_i * 2]
        second_col = fig.axes[row_i * 2 + 1]

        assert first_col.get_ylabel() == expected_label
        assert first_col.yaxis.labelpad == Y_LABEL_PAD
        assert second_col.get_ylabel() == ""

    assert fig.texts == []
    assert fig.subplotpars.hspace == pytest.approx(HSPACE)
    assert fig.subplotpars.wspace == pytest.approx(WSPACE)
    plt.close(fig)


def test_time_varying_verification_uses_two_by_two_grid_for_single_parameter():
    rng = np.random.default_rng(1)
    targets = rng.normal(size=(6, 5, 1))
    estimates = targets[:, None] + rng.normal(scale=0.2, size=(6, 20, 5, 1))

    fig = plot_time_varying_verification(
        estimates,
        targets,
        variable_names=["Drift rate"],
    )

    assert len(fig.axes) == 4
    np.testing.assert_allclose(
        fig.get_size_inches(),
        [BASE_COL_WIDTH * 2, BASE_ROW_HEIGHT * 2 + 0.75],
    )
    assert [ax.get_title() for ax in fig.axes] == [
        "Correlation (Truth vs. Estimate)",
        "NRMSE",
        "Posterior Contraction",
        "Calibration Error",
    ]
    assert [ax.get_ylabel() for ax in fig.axes] == ["Value", "", "Value", ""]
    assert [ax.get_xlabel() for ax in fig.axes] == ["", "", "Step", "Step"]
    assert fig._suptitle.get_text() == "Drift rate"
    plt.close(fig)


def test_joint_prior_positions_legend_and_row_names_without_overlap():
    rng = np.random.default_rng(0)
    names = ["short", "a_much_longer_parameter_name"]
    local_params = {name: rng.normal(size=(5, 10)) for name in names}
    hyper_params = {f"{name}_sigma": rng.normal(size=30) for name in names}

    fig = plot_joint_prior(
        local_params,
        hyper_params,
        {},
        hyper_param_groups={name: [f"{name}_sigma"] for name in names},
        marginal=False,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    grid_axes = fig.axes[:4]

    for row_i, name in enumerate(names):
        panel = grid_axes[row_i * 2]
        label = panel.yaxis.label
        label_bbox = label.get_window_extent(renderer)
        panel_bbox = panel.get_window_extent(renderer)

        assert label.get_text() == name
        assert label.get_fontsize() == 18
        assert (label_bbox.y0 + label_bbox.y1) / 2 == pytest.approx((panel_bbox.y0 + panel_bbox.y1) / 2)
        assert label_bbox.x1 < panel_bbox.x0

    legend_anchor = fig.legends[0].get_bbox_to_anchor().transformed(fig.transFigure.inverted())
    assert fig.subplotpars.hspace == pytest.approx(0.5)
    assert fig.legends[0].get_texts()[0].get_fontsize() == 18
    assert legend_anchor.y0 * fig.get_size_inches()[1] == pytest.approx(0.25)
    plt.close(fig)


def test_plotting_defaults_are_exported():
    assert BASE_COLOR == CATEGORICAL_PALETTE[0]
    assert BASE_COLOR == METRIC_COLORS[0]
    assert BASE_COL_WIDTH > 0
    assert BASE_ROW_HEIGHT > 0
    assert 0 <= DIST_ALPHA <= 1
    assert DIST_ALPHA == pytest.approx(1.0)
    assert OVERLAY_DIST_ALPHA == pytest.approx(0.5)
    assert LABEL_PAD > 0
    assert Y_LABEL_PAD > LABEL_PAD
    assert TITLE_FONTSIZE == 22
    assert LABEL_FONTSIZE == 18
    assert TICK_FONTSIZE == 16
    assert HSPACE == pytest.approx(0.4)
    assert JOINT_HSPACE == pytest.approx(0.5)
    assert WSPACE == pytest.approx(0.2)


def test_diagnostic_plot_font_defaults():
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    expected_font = "Inter" if "Inter" in available_fonts else "DejaVu Sans"
    assert plt.rcParams["font.family"] == [expected_font]
    assert plt.rcParams["mathtext.fontset"] == "cm"


def test_diagnostics_perfect_posterior_has_expected_invariants():
    targets = np.array([[[0.0], [1.0]], [[2.0], [3.0]], [[4.0], [5.0]]])
    estimates = np.repeat(targets[:, None], 4, axis=1)

    np.testing.assert_allclose(correlation_per_step(estimates, targets), 1.0)
    np.testing.assert_allclose(posterior_contraction_per_step(estimates, targets), 1.0)
    np.testing.assert_allclose(nrmse_per_step(estimates, targets), 0.0)
    calibration = calibration_error_per_step(estimates, targets)
    assert calibration.shape == (2, 1)
    assert np.all((calibration >= 0) & (calibration <= 1))


@pytest.mark.parametrize("fn", [correlation_per_step, posterior_contraction_per_step, nrmse_per_step])
def test_diagnostics_reject_non_four_dimensional_estimates(fn):
    with pytest.raises(ValueError, match="shape"):
        fn(np.ones((2, 3, 4)), np.ones((2, 4, 1)))


def test_calibration_rejects_invalid_quantiles():
    with pytest.raises(ValueError, match="Require"):
        calibration_error_per_step(np.ones((2, 3, 1, 1)), np.ones((2, 1, 1)), min_quantile=0.5, max_quantile=0.5)
