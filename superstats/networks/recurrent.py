import bayesflow as bf
import keras

from bayesflow.types import Tensor
from bayesflow.utils import layer_kwargs
from bayesflow.utils.serialization import serializable


@serializable("custom")
class RecurrentNet(bf.networks.SummaryNetwork):
    """
    Implements a simple recurrent network with options compatible with downstream
    bayesflow workflows.

    Parameters
    ----------
    summary_dim : int, optional
        The final output dimensionality. Default is 64.
    hidden_dim : int, optional
        Dimensionality of the hidden state in the recurrent layers. Default is 256.
    recurrent_type : str, optional
        Type of recurrent unit to use. Should correspond to a supported type in `find_recurrent_net`,
        such as "gru" or "lstm". Default is "gru".
    bidirectional : bool, optional
        If True, uses bidirectional wrappers for both recurrent and skip recurrent layers. Default is True.
    dropout : float, optional
        Dropout rate applied within the recurrent layers. Default is 0.05.
    **kwargs
        Additional keyword arguments passed to the parent class constructor.
    """

    def __init__(
        self,
        summary_dim: int = 64,
        hidden_dim: int = 256,
        recurrent_type: str = "lstm",
        bidirectional: bool = True,
        dropout: float = 0.05,
        **kwargs,
    ):

        super().__init__(**layer_kwargs(kwargs))

        if recurrent_type == "lstm":
            recurrent_constructor = keras.layers.LSTM
        elif recurrent_type == "gru":
            recurrent_constructor = keras.layers.GRU
        else:
            raise ValueError(f"recurrent_type must be one of ['lstm', 'gru'], not {recurrent_type}.")

        if bidirectional:
            forward_recurrent = recurrent_constructor(
                units=hidden_dim, 
                dropout=dropout, 
                return_sequences=True
            )
            backward_recurrent = recurrent_constructor(
                units=hidden_dim, 
                dropout=dropout, 
                return_sequences=True
            )

            self.recurrent_forward = forward_recurrent
            self.recurrent_backward = backward_recurrent
        else:
            self.recurrent = recurrent_constructor(units=hidden_dim, dropout=dropout)

        self.summary_stats = keras.layers.Dense(summary_dim)

        self.hidden_dim = hidden_dim
        self.recurrent_type = recurrent_type
        self.bidirectional = bidirectional
        self.dropout = dropout

    def call(self, time_series: Tensor, training: bool = False) -> Tensor:
        if self.bidirectional:
            forward_direct = self.recurrent_forward(time_series, training=training)
            backward_direct = self.recurrent_backward(keras.ops.flip(time_series, axis=1), training=training)
            backward_direct = keras.ops.flip(backward_direct, axis=1)
            out = forward_direct + backward_direct
        else:
            out = self.recurrent(time_series, training=training)

        return self.summary_stats(out)
