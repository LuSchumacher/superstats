from types import MappingProxyType

import keras
import numpy as np
import pytest

from superstats.defaults import (
    DEFAULT_COUPLING_FLOW,
    DEFAULT_CONSISTENCY_MODEL,
    DEFAULT_RECURRENT_NETWORK,
    DEFAULT_TRANSFORMER_NETWORK,
)
from superstats.networks import RecurrentNet
from superstats.simulation.augmentation import (
    ContaminationProcess,
    MissingProcess,
    RandomChoiceContamination,
    RandomMissingProcess,
)
from superstats.utils.dispatch import (
    find_contamination,
    find_inference_network,
    find_missing,
    find_embedding_network,
)


def test_network_defaults_are_frozen():
    assert isinstance(DEFAULT_RECURRENT_NETWORK, MappingProxyType)
    assert isinstance(DEFAULT_TRANSFORMER_NETWORK, MappingProxyType)
    assert isinstance(DEFAULT_COUPLING_FLOW, MappingProxyType)
    assert isinstance(DEFAULT_CONSISTENCY_MODEL, MappingProxyType)

    with pytest.raises(TypeError):
        DEFAULT_RECURRENT_NETWORK["hidden_dim"] = 64


def test_embedding_network_dispatches_recurrent_defaults():
    network = find_embedding_network("recurrent", hidden_dim=16, summary_dim=8)

    assert isinstance(network, RecurrentNet)
    assert network.recurrent_type == "gru"
    assert network.hidden_dim == 16
    assert network.time_axis == 0


def test_embedding_network_dispatches_transformer_defaults():
    network = find_embedding_network("transformer", summary_dim=8, embed_dims=(16,), num_heads=(2,))
    data = np.random.normal(size=(2, 5, 3)).astype("float32")

    out = network(data)

    assert network.time_axis == 0
    assert tuple(out.shape) == (2, 5, 8)


def test_inference_network_dispatches_coupling_defaults():
    network = find_inference_network("coupling")

    assert network.__class__.__name__ == "CouplingFlow"


def test_network_dispatch_passes_existing_layers_through():
    network = keras.layers.Dense(4)

    assert find_embedding_network(network) is network
    assert find_inference_network(network) is network


def test_network_dispatch_rejects_unsupported_inputs():
    with pytest.raises(ValueError):
        find_embedding_network("mlp")
    with pytest.raises(ValueError):
        find_embedding_network("lstm")
    with pytest.raises(TypeError):
        find_embedding_network(RecurrentNet)
    with pytest.raises(TypeError):
        find_embedding_network(None)

    with pytest.raises(ValueError):
        find_inference_network("consistency")
    with pytest.raises(TypeError):
        find_inference_network(None)


def test_missing_dispatches_random_defaults():
    process = find_missing("random")

    assert isinstance(process, RandomMissingProcess)
    assert isinstance(process, MissingProcess)


def test_missing_dispatch_passes_existing_processes_through():
    process = RandomMissingProcess(p_missing=0.0)

    assert find_missing(process) is process
    assert find_missing(None) is None


def test_missing_dispatch_passes_plain_callables_through():
    def missing(data, rng=None):
        return data

    assert find_missing(missing) is missing


def test_contamination_dispatches_random_choice_defaults():
    process = find_contamination("random_choice")

    assert isinstance(process, RandomChoiceContamination)
    assert isinstance(process, ContaminationProcess)


def test_contamination_dispatch_passes_existing_processes_through():
    process = RandomChoiceContamination(p_contaminated=0.0)

    assert find_contamination(process) is process
    assert find_contamination(None) is None


def test_contamination_dispatch_passes_plain_callables_through():
    def contamination(data, rng=None):
        return data

    assert find_contamination(contamination) is contamination


def test_process_dispatch_rejects_unsupported_inputs():
    with pytest.raises(TypeError):
        find_missing("not-callable")
    with pytest.raises(TypeError):
        find_missing(1)

    with pytest.raises(TypeError):
        find_contamination("not-callable")
    with pytest.raises(TypeError):
        find_contamination(1)
