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
        inference_network=flow(),
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
    # Previously tiled targets fail with a clear explanation at the head boundary.
    with pytest.raises(ValueError, match="tiled invariant"):
        workflow.approximator.compute_metrics(
            **workflow.adapter(simulator.sample(batch_size=8, num_steps=3, tile_to_steps=True))
        )


def test_explicit_joint_workflow_uses_the_composites_networks_and_adapter():
    simulator = model()
    approximator = JointApproximator(
        summary_network=keras.layers.Dense(4),
        inference_network=flow(),
        invariant_inference_network=flow(),
        adapter=Workflow.default_adapter(simulator),
        decoder_network=bf.networks.decoders.RecurrentDecoder(hidden_size=4),
    )
    workflow = Workflow(model=simulator, approximator=approximator)
    assert workflow.approximator is approximator
    assert workflow.adapter is approximator.adapter
    assert workflow.embedding_network is approximator.summary_network
    assert workflow.inference_network is approximator.inference_network


def test_invariant_only_model_defaults_to_one_global_summary():
    simulator = Model(
        prior=JointPrior(level=Prior("normal", loc=0, scale=1), offset=Prior("normal", loc=0, scale=1)),
        simulator=lambda level, offset: {"x": level + offset},
        missing=None,
    )
    workflow = Workflow(model=simulator, inference_network=flow())
    assert type(workflow.approximator) is bf.approximators.ContinuousApproximator
    assert workflow.embedding_network.return_sequences is False
    adapted = workflow.adapter(simulator.sample(batch_size=4, num_steps=3))
    assert adapted["inference_variables"].shape == (4, 2)
    workflow.approximator.build(keras.tree.map_structure(np.shape, adapted))
    metrics = workflow.approximator.compute_metrics(**adapted)
    assert np.isfinite(keras.ops.convert_to_numpy(metrics["loss"]))
