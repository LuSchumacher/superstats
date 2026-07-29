from types import SimpleNamespace
from unittest.mock import Mock

import keras
import pytest

from superstats.workflow import Workflow
from superstats.workflow import workflow as workflow_module


class FakeHistory:
    def __init__(self, history):
        self.history = history


@pytest.fixture
def basic_workflow(monkeypatch):
    instances = []

    class FakeBasicWorkflow:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.history = None
            self.approximator = Mock(name="approximator")
            self.fit_offline = Mock(return_value=FakeHistory({"loss": [1.0]}))
            self.fit_online = Mock(return_value=FakeHistory({"loss": [2.0]}))
            instances.append(self)

    monkeypatch.setattr(workflow_module.bf, "BasicWorkflow", FakeBasicWorkflow)
    return instances


def test_init_builds_underlying_workflow_with_embedding_network(basic_workflow):
    adapter = object()
    embedding = keras.layers.Dense(4)
    inference = keras.layers.Dense(2)

    workflow = Workflow(
        adapter=adapter,
        embedding_network=embedding,
        inference_network=inference,
        checkpoint_filepath=None,
    )

    assert workflow.adapter is adapter
    assert workflow.embedding_network is embedding
    assert workflow.inference_network is inference
    assert workflow.workflow.adapter is adapter
    assert workflow.workflow.summary_network is embedding
    assert workflow.workflow.inference_network is inference
    assert workflow.workflow.standardize == "all"


def test_properties_read_and_write_underlying_workflow(basic_workflow):
    workflow = Workflow(adapter=object())
    history = FakeHistory({"loss": [1.0]})
    approximator = object()

    workflow.workflow.history = history
    assert workflow.history is history

    workflow.approximator = approximator
    assert workflow.approximator is approximator


def test_fit_offline_delegates_and_saves_history(basic_workflow, tmp_path):
    workflow = Workflow(
        adapter=object(),
        checkpoint_filepath=str(tmp_path),
        restore_approximator=False,
        restore_history=False,
    )
    data = {"x": [1]}
    validation_data = {"x": [2]}

    history = workflow.fit_offline(data, validation_data, epochs=3, batch_size=5)

    assert history.history == {"loss": [1.0]}
    workflow.workflow.fit_offline.assert_called_once_with(
        data=data, epochs=3, batch_size=5, validation_data=validation_data
    )
    assert workflow.history is history
    assert (tmp_path / "history.pkl").exists()


def test_fit_online_restores_sampler_after_training(basic_workflow):
    simulator = SimpleNamespace(sample=Mock(name="sample"))
    workflow = Workflow(simulator=simulator, adapter=object())
    original_sample = simulator.sample

    workflow.fit_online(num_steps=7, epochs=2, num_batches_per_epoch=3, batch_size=4, save_history=False)

    assert simulator.sample is original_sample
    workflow.workflow.fit_online.assert_called_once_with(epochs=2, num_batches_per_epoch=3, batch_size=4)
    simulator.sample.assert_not_called()


def test_fit_online_restores_sampler_when_training_fails(basic_workflow):
    simulator = SimpleNamespace(sample=Mock(name="sample"))
    workflow = Workflow(simulator=simulator, adapter=object())
    original_sample = simulator.sample
    workflow.workflow.fit_online.side_effect = RuntimeError("training failed")

    with pytest.raises(RuntimeError, match="training failed"):
        workflow.fit_online(num_steps=4, save_history=False)

    assert simulator.sample is original_sample
