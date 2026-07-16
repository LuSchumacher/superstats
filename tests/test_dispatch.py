from types import MappingProxyType

import keras
import numpy as np
import pytest

from superstats.defaults import (
    DEFAULT_INFERENCE_NETWORK,
    DEFAULT_RECURRENT_SUMMARY_NETWORK,
    DEFAULT_TRANSFORMER_SUMMARY_NETWORK,
)
from superstats.networks import RecurrentNet
from superstats.utils.dispatch import find_inference_network, find_summary_network


def test_network_defaults_are_frozen():
    assert isinstance(DEFAULT_RECURRENT_SUMMARY_NETWORK, MappingProxyType)
    assert isinstance(DEFAULT_TRANSFORMER_SUMMARY_NETWORK, MappingProxyType)
    assert isinstance(DEFAULT_INFERENCE_NETWORK, MappingProxyType)

    with pytest.raises(TypeError):
        DEFAULT_RECURRENT_SUMMARY_NETWORK["hidden_dim"] = 64


def test_summary_network_dispatches_recurrent_defaults():
    network = find_summary_network("recurrent", hidden_dim=16, summary_dim=8)

    assert isinstance(network, RecurrentNet)
    assert network.recurrent_type == "lstm"
    assert network.hidden_dim == 16
    assert network.time_axis == 0


def test_summary_network_dispatches_transformer_defaults():
    network = find_summary_network("transformer", summary_dim=8, embed_dims=(16,), num_heads=(2,))
    data = np.random.normal(size=(2, 5, 3)).astype("float32")

    out = network(data)

    assert network.time_axis == 0
    assert tuple(out.shape) == (2, 5, 8)


def test_inference_network_dispatches_coupling_defaults():
    network = find_inference_network("coupling")

    assert network.__class__.__name__ == "CouplingFlow"


def test_network_dispatch_passes_existing_layers_through():
    network = keras.layers.Dense(4)

    assert find_summary_network(network) is network
    assert find_inference_network(network) is network


def test_network_dispatch_rejects_unsupported_inputs():
    with pytest.raises(ValueError):
        find_summary_network("mlp")
    with pytest.raises(TypeError):
        find_summary_network(RecurrentNet)
    with pytest.raises(TypeError):
        find_summary_network(None)

    with pytest.raises(ValueError):
        find_inference_network("consistency")
    with pytest.raises(TypeError):
        find_inference_network(None)
