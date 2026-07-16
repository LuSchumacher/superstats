import numpy as np
import pytest
import keras

from superstats.networks import RecurrentNet
from tests.utils import assert_layers_equal

BATCH_SIZE = 4
SEQ_LEN = 10
NUM_FEATURES = 3


@pytest.mark.parametrize("recurrent_type", ["lstm", "gru"])
def test_recurrent_net_unidirectional_output_shape(recurrent_type):
    net = RecurrentNet(
        summary_dim=8,
        hidden_dim=16,
        recurrent_type=recurrent_type,
        bidirectional=False,
    )
    x = np.random.normal(size=(BATCH_SIZE, SEQ_LEN, NUM_FEATURES)).astype("float32")

    out = net(x)

    assert tuple(out.shape) == (BATCH_SIZE, SEQ_LEN, 8)


@pytest.mark.parametrize("recurrent_type", ["lstm", "gru"])
def test_recurrent_net_bidirectional_output_shape(recurrent_type):
    net = RecurrentNet(
        summary_dim=8,
        hidden_dim=16,
        recurrent_type=recurrent_type,
        bidirectional=True,
    )
    x = np.random.normal(size=(BATCH_SIZE, SEQ_LEN, NUM_FEATURES)).astype("float32")

    out = net(x)

    assert tuple(out.shape) == (BATCH_SIZE, SEQ_LEN, 8)


def test_recurrent_net_invalid_recurrent_type_raises():
    with pytest.raises(ValueError):
        RecurrentNet(recurrent_type="not-a-type")


def test_recurrent_net_defaults_to_sum_merge_mode():
    net = RecurrentNet(bidirectional=True)

    assert net.recurrent_layers[0].merge_mode == "sum"


def test_recurrent_net_custom_merge_mode():
    net = RecurrentNet(bidirectional=True, merge_mode="concat")

    assert net.recurrent_layers[0].merge_mode == "concat"


def test_recurrent_net_invalid_merge_mode_raises():
    with pytest.raises(ValueError):
        RecurrentNet(merge_mode="invalid")


def test_recurrent_net_layer_norm_can_be_disabled():
    net = RecurrentNet(layer_norm=False)

    assert len(net.normalization_layers) == len(net.recurrent_layers)
    assert all(layer is None for layer in net.normalization_layers)


def test_recurrent_net_adds_normalization_per_recurrent_layer():
    net = RecurrentNet(
        hidden_dim=(16, 8),
        recurrent_type=("lstm", "gru"),
        bidirectional=(False, True),
        layer_norm=(True, False),
    )

    assert isinstance(net.normalization_layers[0], keras.layers.LayerNormalization)
    assert net.normalization_layers[1] is None


@pytest.mark.parametrize("recurrent_type", ["lstm", "gru"])
@pytest.mark.parametrize("bidirectional", [False, True])
def test_recurrent_net_save_and_load(tmp_path, recurrent_type, bidirectional):
    net = RecurrentNet(
        summary_dim=8,
        hidden_dim=16,
        recurrent_type=recurrent_type,
        bidirectional=bidirectional,
        dropout=0.0,
    )
    x = np.random.normal(size=(BATCH_SIZE, SEQ_LEN, NUM_FEATURES)).astype("float32")

    net(x)  # build
    keras.saving.save_model(net, tmp_path / "recurrent_net.keras")
    loaded = keras.saving.load_model(tmp_path / "recurrent_net.keras")

    assert_layers_equal(net, loaded)
