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
