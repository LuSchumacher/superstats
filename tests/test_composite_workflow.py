"""Exercise untiled targets through the actual BayesFlow workflow trainer."""

import bayesflow as bf
import keras
import numpy as np
import pytest

from superstats import JointEstimator, JointPrior, MarginalEstimator, Model, Prior, Workflow
from superstats.transition import RandomWalk


def model():
    return Model(
        prior=JointPrior(
            theta=RandomWalk(sigma=Prior("halfnormal", scale=0.1), delta=0.0),
            level=Prior("normal", loc=0.0, scale=1.0),
        ),
        simulator=lambda theta, level: {"x": theta + level},
        missing=None,
    )


def flow():
    return bf.networks.CouplingFlow(depth=1, subnet_kwargs={"widths": (8,)})


def test_default_workflow_trains_samples_and_scores_separate_targets():
    simulator = model()
    workflow = Workflow(
        model=simulator,
        embedding_network=keras.layers.Dense(4),
        varying_inference_network=flow(),
        invariant_inference_network=flow(),
    )
    assert isinstance(workflow.estimator, MarginalEstimator)
    data = simulator.sample(batch_size=8, num_steps=3)
    adapted = workflow.adapter(data)
    assert adapted["inference_variables"].shape == (8, 3, 1)
    assert adapted["invariant_variables"].shape == (8, 2)
    history = workflow.fit_offline(data, data, epochs=1, batch_size=4, verbose=0)
    assert {"invariant/loss", "varying/loss", "val_loss"} <= history.history.keys()
    samples = workflow.sample(data, num_samples=3, batch_size=4, seed=7)
    assert samples["theta"].shape == (8, 3, 3, 1)
    for key in simulator.hyper_keys + simulator.shared_keys:
        assert samples[key].shape == (8, 3, 1)
    density = workflow.estimator.log_prob(data)
    assert density.shape == (8,)
    assert np.isfinite(density).all()


@pytest.mark.parametrize(
    ("name", "expected"),
    [("marginal", MarginalEstimator), ("joint", JointEstimator)],
)
@pytest.mark.parametrize("mode", ["smoothing", "filtering"])
def test_workflow_selects_composite_and_mode_by_name(name, expected, mode):
    workflow = Workflow(model=model(), estimator=name, mode=mode)

    assert isinstance(workflow.estimator, expected)
    assert workflow.mode == mode
    assert workflow.estimator.mode == mode
    for network in (workflow.varying_inference_network, workflow.invariant_inference_network):
        assert network.get_config()["depth"] == 2
        assert network.get_config()["transform"] == "spline"

    summary = workflow.estimator.summary_network
    assert isinstance(summary, bf.networks.RecurrentNetwork)
    assert summary.bidirectional is (mode == "smoothing")

    if name == "joint":
        encoder = workflow.estimator.sequence_estimator.encoder_network
        assert isinstance(encoder, keras.layers.Identity)
        decoder_type = (
            bf.networks.decoders.TransformerDecoder if mode == "smoothing" else bf.networks.decoders.RecurrentDecoder
        )
        assert isinstance(workflow.estimator.decoder_network, decoder_type)


@pytest.mark.parametrize("cls", [MarginalEstimator, JointEstimator])
@pytest.mark.parametrize("mode", ["smoothing", "filtering"])
def test_external_custom_estimator_takes_precedence(cls, mode):
    external_model = model()
    summary_network = keras.layers.Dense(4)
    varying_inference_network = flow()
    invariant_inference_network = flow()
    kwargs = {}
    if cls is JointEstimator:
        kwargs = {
            "decoder_network": bf.networks.decoders.RecurrentDecoder(hidden_size=4),
        }

    estimator = cls(
        summary_network=summary_network,
        varying_inference_network=varying_inference_network,
        invariant_inference_network=invariant_inference_network,
        mode=mode,
        adapter=Workflow.default_adapter(external_model),
        **kwargs,
    )
    estimator.model = external_model
    ignored_mode = "filtering" if mode == "smoothing" else "smoothing"
    workflow = Workflow(model=model(), adapter=object(), estimator=estimator, mode=ignored_mode)

    assert workflow.estimator is estimator
    assert workflow.mode == mode
    assert workflow.model is external_model
    assert workflow.adapter is estimator.adapter
    assert workflow.embedding_network is summary_network
    assert workflow.varying_inference_network is varying_inference_network
    assert workflow.invariant_inference_network is invariant_inference_network


def test_workflow_rejects_unknown_composite_names_and_modes():
    with pytest.raises(ValueError, match="estimator"):
        Workflow(model=model(), estimator="mixture")
    with pytest.raises(ValueError, match="mode"):
        Workflow(model=model(), estimator="marginal", mode="prediction")


@pytest.mark.parametrize(
    ("argument", "value", "replacement"),
    [("approximator", "joint", "estimator"), ("restore_approximator", False, "restore_estimator")],
)
def test_workflow_rejects_renamed_keywords(argument, value, replacement):
    with pytest.raises(TypeError, match=rf"Use {replacement}\b"):
        Workflow(model=model(), **{argument: value})


def test_invariant_only_model_defaults_to_one_global_summary():
    simulator = Model(
        prior=JointPrior(level=Prior("normal", loc=0, scale=1), offset=Prior("normal", loc=0, scale=1)),
        simulator=lambda level, offset: {"x": level + offset},
        missing=None,
    )
    workflow = Workflow(model=simulator, invariant_inference_network=flow())
    assert isinstance(workflow.estimator, MarginalEstimator)
    assert workflow.varying_inference_network is None
    adapted = workflow.adapter(simulator.sample(batch_size=4, num_steps=3))
    assert adapted["invariant_variables"].shape == (4, 2)
    workflow.estimator.build(keras.tree.map_structure(np.shape, adapted))
    metrics = workflow.estimator.compute_metrics(**adapted)
    assert np.isfinite(keras.ops.convert_to_numpy(metrics["loss"]))


@pytest.mark.parametrize("name", ["marginal", "joint"])
@pytest.mark.parametrize("group", ["varying", "invariant"])
def test_single_group_workflow_trains_and_samples(name, group):
    if group == "varying":
        simulator = Model(
            prior=JointPrior(theta=RandomWalk(sigma=0.1, delta=0.0)),
            simulator=lambda theta: {"x": theta},
            missing=None,
        )
    else:
        simulator = Model(
            prior=JointPrior(level=Prior("normal", loc=0, scale=1)),
            simulator=lambda level: {"x": level},
            missing=None,
        )
    cls = MarginalEstimator if name == "marginal" else JointEstimator
    kwargs = {}
    if name == "joint" and group == "varying":
        kwargs["decoder_network"] = bf.networks.decoders.RecurrentDecoder(embed_dim=4, output_dim=4)
    estimator = cls(
        adapter=Workflow.default_adapter(simulator),
        has_varying=group == "varying",
        has_invariant=group == "invariant",
        varying_inference_network=flow() if group == "varying" else "coupling",
        invariant_inference_network=flow() if group == "invariant" else "coupling",
        summary_network=keras.layers.Dense(4),
        **kwargs,
    )
    workflow = Workflow(model=simulator, estimator=estimator)
    data = simulator.sample(batch_size=4, num_steps=3)
    history = workflow.fit_offline(data, data, epochs=1, batch_size=4, verbose=0)
    assert np.isfinite(history.history["loss"]).all()
    samples = workflow.sample(data, num_samples=2, seed=7)
    key = "theta" if group == "varying" else "level"
    assert set(samples) == {key}
    assert samples[key].shape == ((4, 2, 3, 1) if group == "varying" else (4, 2, 1))
    assert np.isfinite(workflow.estimator.log_prob(data)).all()


@pytest.mark.parametrize("name", [None, ""])
def test_workflow_rejects_implicit_estimator_selection(name):
    with pytest.raises((TypeError, ValueError), match="estimator"):
        Workflow(model=model(), estimator=name)


def test_workflow_rejects_network_for_absent_target_group():
    simulator = Model(
        prior=JointPrior(level=Prior("normal", loc=0, scale=1)),
        simulator=lambda level: {"x": level},
        missing=None,
    )
    with pytest.raises(ValueError, match="varying_inference_network"):
        Workflow(model=simulator, varying_inference_network=flow())
    with pytest.raises(TypeError, match="varying_inference_network"):
        Workflow(model=simulator, inference_network=flow())


@pytest.mark.parametrize("name", ["marginal", "joint"])
@pytest.mark.parametrize("group", ["varying", "invariant"])
def test_string_selected_workflow_activates_available_targets(name, group):
    prior = (
        JointPrior(theta=RandomWalk(sigma=0.1, delta=0.0))
        if group == "varying"
        else JointPrior(theta=Prior("normal", loc=0, scale=1))
    )
    simulator = Model(prior=prior, simulator=lambda theta: {"x": theta}, missing=None)
    workflow = Workflow(model=simulator, estimator=name)
    assert workflow.estimator.has_varying is (group == "varying")
    assert workflow.estimator.has_invariant is (group == "invariant")
    network = workflow.varying_inference_network if group == "varying" else workflow.invariant_inference_network
    assert network.get_config()["depth"] == 2
    assert network.get_config()["transform"] == "spline"


@pytest.mark.parametrize("argument", ["embedding_network", "varying_inference_network", "invariant_inference_network"])
def test_workflow_rejects_none_network_selection(argument):
    with pytest.raises(TypeError, match="network"):
        Workflow(model=model(), **{argument: None})
