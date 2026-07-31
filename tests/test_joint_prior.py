import inspect

import matplotlib.pyplot as plt
import numpy as np
import pytest

import superstats.prior.joint_prior as joint_prior_module
from superstats.defaults import BASE_COL_WIDTH
from superstats.prior import JointPrior, Prior
from superstats.transition.stochastic import RandomWalk

BATCH_SIZE = 6
NUM_STEPS = 10


def _build_joint_prior():
    return JointPrior(
        v=RandomWalk(),
        a=Prior("halfnormal", scale=1.0),
        tau=0.2,
        bias=0,
    )


def test_joint_prior_sample_groups_and_shapes():
    prior = _build_joint_prior()
    result = prior.sample(batch_size=BATCH_SIZE, num_steps=NUM_STEPS)

    assert set(result.keys()) == {
        "local_params",
        "deterministic_params",
        "hyper_params",
        "shared_params",
        "fixed_params",
    }

    assert result["local_params"]["v"].shape == (BATCH_SIZE, NUM_STEPS)
    assert result["deterministic_params"] == {}
    assert "v_sigma" in result["hyper_params"]
    assert result["hyper_params"]["v_sigma"].shape == (BATCH_SIZE,)

    assert result["shared_params"]["a"].shape == (BATCH_SIZE,)

    assert result["fixed_params"]["tau"] == pytest.approx(0.2)
    assert result["fixed_params"]["bias"] == 0
    assert isinstance(result["fixed_params"]["bias"], int)


def test_joint_prior_rejects_non_positive_batch_size():
    prior = _build_joint_prior()
    with pytest.raises(ValueError):
        prior.sample(batch_size=0, num_steps=NUM_STEPS)


def test_joint_prior_rejects_non_positive_num_steps():
    prior = _build_joint_prior()
    with pytest.raises(ValueError):
        prior.sample(batch_size=BATCH_SIZE, num_steps=0)


def test_joint_prior_rejects_unknown_parameter_type():
    prior = JointPrior(x=[1, 2, 3])
    with pytest.raises(TypeError):
        prior.sample(batch_size=BATCH_SIZE, num_steps=NUM_STEPS)


def test_joint_prior_param_bounds_collects_transition_bounds():
    prior = _build_joint_prior()
    bounds = prior._param_bounds()
    assert "v" in bounds
    np.testing.assert_allclose(bounds["v"], prior.params["v"].bounds)


def test_joint_prior_plot_methods_have_no_var_keyword_arguments():
    for method in (
        JointPrior.plot_time_varying_prior,
        JointPrior.plot_time_invariant_prior,
        JointPrior.plot_joint_prior,
    ):
        assert all(
            parameter.kind is not inspect.Parameter.VAR_KEYWORD
            for parameter in inspect.signature(method).parameters.values()
        )


def test_joint_prior_time_invariant_plot_uses_three_columns_for_six_params():
    prior = JointPrior(**{f"p{i}": Prior("normal") for i in range(6)})

    fig = prior.plot_time_invariant_prior(num_draws=30, dist_type="hist")

    assert len(fig.axes) == 6
    assert fig.get_size_inches()[0] == pytest.approx(BASE_COL_WIDTH * 3)
    plt.close(fig)


def test_joint_prior_time_varying_plot_accepts_default_distribution_alpha():
    prior = _build_joint_prior()

    assert inspect.signature(JointPrior.plot_time_varying_prior).parameters["dist_alpha"].default == 1.0

    fig = prior.plot_time_varying_prior(
        num_steps=8,
        num_trajectories=4,
    )

    assert fig is not None
    assert any(ax.patches for ax in fig.axes)
    plt.close(fig)


def test_joint_prior_time_varying_plot_forwards_all_arguments(monkeypatch):
    captured = {}
    sentinel = object()

    def fake_plot(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(joint_prior_module, "plot_time_varying_prior", fake_plot)
    prior = _build_joint_prior()

    result = prior.plot_time_varying_prior(
        num_steps=8,
        num_trajectories=4,
        num_cols=1,
        marginal=False,
        dist_type="kde",
        num_bins=13,
        dist_alpha=0.35,
        alpha=0.2,
        color="red",
        title_fontsize=11,
        label_fontsize=12,
        tick_fontsize=9,
        figsize=(5.0, 4.0),
    )

    assert result is sentinel
    assert captured["num_cols"] == 1
    assert captured["marginal"] is False
    assert captured["dist_type"] == "kde"
    assert captured["num_bins"] == 13
    assert captured["dist_alpha"] == 0.35
    assert captured["alpha"] == 0.2
    assert captured["color"] == "red"
    assert captured["title_fontsize"] == 11
    assert captured["label_fontsize"] == 12
    assert captured["tick_fontsize"] == 9
    assert "hspace" not in captured
    assert "wspace" not in captured
    assert captured["figsize"] == (5.0, 4.0)


def test_joint_prior_time_invariant_plot_forwards_all_arguments(monkeypatch):
    captured = {}
    sentinel = object()

    def fake_plot(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(joint_prior_module, "plot_time_invariant_prior", fake_plot)
    prior = _build_joint_prior()

    result = prior.plot_time_invariant_prior(
        num_draws=10,
        dist_type="kde",
        num_bins=13,
        dist_alpha=0.35,
        color="red",
        num_cols=3,
        title_fontsize=11,
        label_fontsize=12,
        tick_fontsize=9,
        figsize=(5.0, 4.0),
    )

    assert result is sentinel
    assert captured["dist_type"] == "kde"
    assert captured["num_bins"] == 13
    assert captured["dist_alpha"] == 0.35
    assert captured["color"] == "red"
    assert captured["num_cols"] == 3
    assert captured["title_fontsize"] == 11
    assert captured["label_fontsize"] == 12
    assert captured["tick_fontsize"] == 9
    assert "hspace" not in captured
    assert "wspace" not in captured
    assert captured["figsize"] == (5.0, 4.0)


def test_joint_prior_joint_plot_forwards_all_arguments(monkeypatch):
    captured = {}
    sentinel = object()

    def fake_plot(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(joint_prior_module, "plot_joint_prior", fake_plot)
    prior = _build_joint_prior()

    result = prior.plot_joint_prior(
        num_steps=8,
        num_trajectories=4,
        num_draws=10,
        marginal=False,
        dist_type="kde",
        num_bins=13,
        dist_alpha=0.35,
        color="red",
        title_fontsize=11,
        label_fontsize=12,
        tick_fontsize=9,
        alpha=0.2,
        figsize=(5.0, 4.0),
    )

    assert result is sentinel
    assert captured["marginal"] is False
    assert captured["dist_type"] == "kde"
    assert captured["num_bins"] == 13
    assert captured["dist_alpha"] == 0.35
    assert captured["color"] == "red"
    assert captured["title_fontsize"] == 11
    assert captured["label_fontsize"] == 12
    assert captured["tick_fontsize"] == 9
    assert captured["alpha"] == 0.2
    assert "hspace" not in captured
    assert "wspace" not in captured
    assert captured["figsize"] == (5.0, 4.0)
