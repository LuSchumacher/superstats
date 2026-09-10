import inspect

import matplotlib.pyplot as plt
import numpy as np
import pytest

import superstats.prior.joint_prior as joint_prior_module
from superstats.defaults import BASE_COL_WIDTH
from superstats.prior import JointPrior, Prior
from superstats.simulation import Model
from superstats.transition import (
    AutoRegression,
    Exponential,
    GaussianProcess,
    Jump,
    LevyFlight,
    Linear,
    Logarithmic,
    Mixture,
    OrnsteinUhlenbeck,
    Polynomial,
    RandomWalk,
)

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


STOCHASTIC_TRANSITION_CATEGORY_CASES = [
    pytest.param(
        lambda: RandomWalk(sigma=Prior("halfnormal", scale=0.1), delta=0.0),
        {"sigma"},
        {"delta"},
        id="random-walk",
    ),
    pytest.param(
        lambda: AutoRegression(sigma=Prior("halfnormal", scale=0.1), phi=0.9, delta=0.0),
        {"sigma"},
        {"phi", "delta"},
        id="auto-regression",
    ),
    pytest.param(
        lambda: OrnsteinUhlenbeck(sigma=Prior("halfnormal", scale=0.1), mu=0.0, theta=0.1),
        {"sigma"},
        {"mu", "theta"},
        id="ornstein-uhlenbeck",
    ),
    pytest.param(
        lambda: LevyFlight(
            sigma=Prior("halfnormal", scale=0.1),
            delta=0.0,
            alpha=1.5,
            beta=0.0,
        ),
        {"sigma"},
        {"delta", "alpha", "beta"},
        id="levy-flight",
    ),
    pytest.param(
        lambda: Jump(p_jump=Prior("beta", a=2, b=2)),
        {"p_jump"},
        set(),
        id="jump",
    ),
    pytest.param(
        lambda: GaussianProcess(
            kernel_params={
                "length_scale": Prior("halfnormal", scale=0.5),
                "amplitude": 1.0,
            }
        ),
        {"length_scale"},
        {"amplitude"},
        id="gaussian-process",
    ),
    pytest.param(
        lambda: Mixture(
            transitions=[RandomWalk(), Jump()],
            mixture_weights=(0.5, 0.5),
        ),
        {"rw_sigma"},
        {"rw_delta", "jump_p_jump", "mixture_weights"},
        id="mixture",
    ),
]


DETERMINISTIC_TRANSITION_CATEGORY_CASES = [
    pytest.param(
        lambda: Linear(intercept=Prior("normal"), beta=0.0),
        {"intercept"},
        {"beta"},
        id="linear",
    ),
    pytest.param(
        lambda: Polynomial(intercept=Prior("normal"), betas=[0.0, 0.0], degree=2),
        {"intercept"},
        {"beta_1", "beta_2"},
        id="polynomial",
    ),
    pytest.param(
        lambda: Exponential(intercept=Prior("normal"), beta=0.0),
        {"intercept"},
        {"beta"},
        id="exponential",
    ),
    pytest.param(
        lambda: Logarithmic(intercept=Prior("normal"), beta=0.0),
        {"intercept"},
        {"beta"},
        id="logarithmic",
    ),
]


def _assert_parameter_categories(model, expected):
    actual = {
        "local_params": set(model.local_keys),
        "deterministic_params": set(model.deterministic_keys),
        "hyper_params": set(model.hyper_keys),
        "shared_params": set(model.shared_keys),
        "fixed_params": set(model.fixed_keys),
    }
    assert actual == expected
    all_keys = [key for keys in actual.values() for key in keys]
    assert len(all_keys) == len(set(all_keys)), "A parameter was registered in more than one category."


@pytest.mark.parametrize("transition_factory, hyper_names, fixed_names", STOCHASTIC_TRANSITION_CATEGORY_CASES)
def test_every_stochastic_transition_uses_exact_model_parameter_categories(
    transition_factory,
    hyper_names,
    fixed_names,
):
    prior = JointPrior(theta=transition_factory(), shared=Prior("normal"), fixed=0.25)

    def simulator(theta, shared, fixed):
        return {"observation": theta + shared + fixed}

    model = Model(prior=prior, simulator=simulator, missing=None)

    _assert_parameter_categories(
        model,
        {
            "local_params": {"theta"},
            "deterministic_params": set(),
            "hyper_params": {f"theta_{name}" for name in hyper_names},
            "shared_params": {"shared"},
            "fixed_params": {"fixed", *(f"theta_{name}" for name in fixed_names)},
        },
    )


@pytest.mark.parametrize("transition_factory, hyper_names, fixed_names", DETERMINISTIC_TRANSITION_CATEGORY_CASES)
def test_every_deterministic_transition_uses_exact_model_parameter_categories(
    transition_factory,
    hyper_names,
    fixed_names,
):
    prior = JointPrior(theta=transition_factory(), shared=Prior("normal"), fixed=0.25)

    def simulator(theta, shared, fixed):
        return {"observation": theta + shared + fixed}

    model = Model(prior=prior, simulator=simulator, missing=None)

    _assert_parameter_categories(
        model,
        {
            "local_params": set(),
            "deterministic_params": {"theta"},
            "hyper_params": {f"theta_{name}" for name in hyper_names},
            "shared_params": {"shared"},
            "fixed_params": {"fixed", *(f"theta_{name}" for name in fixed_names)},
        },
    )


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
