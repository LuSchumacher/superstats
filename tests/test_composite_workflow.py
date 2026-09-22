"""Exercise untiled targets through the actual BayesFlow workflow trainer."""

import bayesflow as bf
import keras
import numpy as np
import pytest

from superstats import JointApproximator, JointPrior, MarginalApproximator, Model, Prior, Workflow
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
    assert isinstance(workflow.approximator, MarginalApproximator)
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
    density = workflow.approximator.log_prob(data)
    assert density.shape == (8,)
    assert np.isfinite(density).all()


@pytest.mark.parametrize(
    ("name", "expected"),
    [("marginal", MarginalApproximator), ("joint", JointApproximator)],
)
@pytest.mark.parametrize("mode", ["smoothing", "filtering"])
def test_workflow_selects_composite_and_mode_by_name(name, expected, mode):
    workflow = Workflow(model=model(), approximator=name, mode=mode)

    assert isinstance(workflow.approximator, expected)
    assert workflow.mode == mode
    assert workflow.approximator.mode == mode
    for network in (workflow.varying_inference_network, workflow.invariant_inference_network):
        assert network.get_config()["depth"] == 2
        assert network.get_config()["transform"] == "spline"

    summary = workflow.approximator.summary_network
    assert isinstance(summary, bf.networks.RecurrentNetwork)
    assert summary.bidirectional is (mode == "smoothing")

    if name == "joint":
        encoder = workflow.approximator.sequence_approximator.encoder_network
        assert isinstance(encoder, keras.layers.Identity)
        decoder_type = (
            bf.networks.decoders.TransformerDecoder if mode == "smoothing" else bf.networks.decoders.RecurrentDecoder
        )
        assert isinstance(workflow.approximator.decoder_network, decoder_type)


@pytest.mark.parametrize("cls", [MarginalApproximator, JointApproximator])
@pytest.mark.parametrize("mode", ["smoothing", "filtering"])
def test_external_custom_approximator_takes_precedence(cls, mode):
    external_model = model()
    summary_network = keras.layers.Dense(4)
    varying_inference_network = flow()
    invariant_inference_network = flow()
    kwargs = {}
    if cls is JointApproximator:
        kwargs = {
            "decoder_network": bf.networks.decoders.RecurrentDecoder(hidden_size=4),
        }

    approximator = cls(
        summary_network=summary_network,
        varying_inference_network=varying_inference_network,
        invariant_inference_network=invariant_inference_network,
        mode=mode,
        adapter=Workflow.default_adapter(external_model),
        **kwargs,
    )
    approximator.model = external_model
    ignored_mode = "filtering" if mode == "smoothing" else "smoothing"
    workflow = Workflow(model=model(), adapter=object(), approximator=approximator, mode=ignored_mode)

    assert workflow.approximator is approximator
    assert workflow.mode == mode
    assert workflow.model is external_model
    assert workflow.adapter is approximator.adapter
    assert workflow.embedding_network is summary_network
    assert workflow.varying_inference_network is varying_inference_network
    assert workflow.invariant_inference_network is invariant_inference_network


def test_workflow_rejects_unknown_composite_names_and_modes():
    with pytest.raises(ValueError, match="approximator"):
        Workflow(model=model(), approximator="mixture")
    with pytest.raises(ValueError, match="mode"):
        Workflow(model=model(), approximator="marginal", mode="prediction")


def test_invariant_only_model_defaults_to_one_global_summary():
    simulator = Model(
        prior=JointPrior(level=Prior("normal", loc=0, scale=1), offset=Prior("normal", loc=0, scale=1)),
        simulator=lambda level, offset: {"x": level + offset},
        missing=None,
    )
    workflow = Workflow(model=simulator, invariant_inference_network=flow())
    assert isinstance(workflow.approximator, MarginalApproximator)
    assert workflow.varying_inference_network is None
    adapted = workflow.adapter(simulator.sample(batch_size=4, num_steps=3))
    assert adapted["invariant_variables"].shape == (4, 2)
    workflow.approximator.build(keras.tree.map_structure(np.shape, adapted))
    metrics = workflow.approximator.compute_metrics(**adapted)
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
    cls = MarginalApproximator if name == "marginal" else JointApproximator
    kwargs = {}
    if name == "joint" and group == "varying":
        kwargs["decoder_network"] = bf.networks.decoders.RecurrentDecoder(embed_dim=4, output_dim=4)
    approximator = cls(
        adapter=Workflow.default_adapter(simulator),
        has_varying=group == "varying",
        has_invariant=group == "invariant",
        varying_inference_network=flow() if group == "varying" else "coupling",
        invariant_inference_network=flow() if group == "invariant" else "coupling",
        summary_network=keras.layers.Dense(4),
        **kwargs,
    )
    workflow = Workflow(model=simulator, approximator=approximator)
    data = simulator.sample(batch_size=4, num_steps=3)
    history = workflow.fit_offline(data, data, epochs=1, batch_size=4, verbose=0)
    assert np.isfinite(history.history["loss"]).all()
    samples = workflow.sample(data, num_samples=2, seed=7)
    key = "theta" if group == "varying" else "level"
    assert set(samples) == {key}
    assert samples[key].shape == ((4, 2, 3, 1) if group == "varying" else (4, 2, 1))
    assert np.isfinite(workflow.approximator.log_prob(data)).all()


@pytest.mark.parametrize("name", [None, ""])
def test_workflow_rejects_implicit_approximator_selection(name):
    with pytest.raises((TypeError, ValueError), match="approximator"):
        Workflow(model=model(), approximator=name)


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
    workflow = Workflow(model=simulator, approximator=name)
    assert workflow.approximator.has_varying is (group == "varying")
    assert workflow.approximator.has_invariant is (group == "invariant")
    network = workflow.varying_inference_network if group == "varying" else workflow.invariant_inference_network
    assert network.get_config()["depth"] == 2
    assert network.get_config()["transform"] == "spline"


@pytest.mark.parametrize("argument", ["embedding_network", "varying_inference_network", "invariant_inference_network"])
def test_workflow_rejects_none_network_selection(argument):
    with pytest.raises(TypeError, match="network"):
        Workflow(model=model(), **{argument: None})
