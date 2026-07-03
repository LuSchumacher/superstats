import bayesflow as bf
import keras

from bayesflow.types import Tensor
from bayesflow.utils import layer_kwargs
from bayesflow.utils.serialization import serializable


@serializable("custom")
class RecurrentNet(bf.networks.SummaryNetwork):
    """Implements a simple recurrent network with options compatible with downstream
    bayesflow workflows.

    Parameters
    ----------
    summary_dim    : int, optional, default: 64
        The final output dimensionality.
    hidden_dim     : int, optional, default: 256
        Dimensionality of the hidden state in the recurrent layers.
    recurrent_type : {"lstm", "gru"}, optional, default: "lstm"
        Type of recurrent unit to use.
    bidirectional  : bool, optional, default: True
        If True, the sequence is processed by separate forward and
        backward recurrent layers and their outputs are summed. If
        False, a single recurrent layer processes the sequence forward
        only, and only its final hidden state is used.
    dropout        : float in [0, 1], optional, default: 0.05
        Dropout rate applied within the recurrent layers.
    **kwargs
        Additional keyword arguments passed to the parent class
        constructor.

    Notes
    -----
    When `bidirectional=True`, the backward pass runs the recurrent
    layer over the time-reversed sequence and flips its output back
    before summing with the forward pass. Both layers are built with
    `return_sequences=True` (required so the two directions can be
    summed per-timestep), so the projection is applied per-timestep
    and the output retains the sequence-length axis. When
    `bidirectional=False`, only the final hidden state of a single
    recurrent layer is projected, giving a fixed-size vector.

    Raises
    ------
    ValueError
        If `recurrent_type` is not "lstm" or "gru".
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
        """Compute summary statistics for a batch of time series.

        Parameters
        ----------
        time_series : Tensor of shape (batch_size, sequence_length, num_features)
            Input time series.
        training    : bool, optional, default: False
            Whether the layer is in training mode (affects dropout).

        Returns
        -------
        summary : Tensor - the learned summary. Shape is
            (batch_size, summary_dim) if `bidirectional=False`, or
            (batch_size, sequence_length, summary_dim) if
            `bidirectional=True` (see class Notes — the per-timestep
            shape in the bidirectional case may not be intended for a
            `SummaryNetwork`, which is normally expected to return a
            fixed-size embedding).
        """
        if self.bidirectional:
            forward_direct = self.recurrent_forward(time_series, training=training)
            backward_direct = self.recurrent_backward(keras.ops.flip(time_series, axis=1), training=training)
            backward_direct = keras.ops.flip(backward_direct, axis=1)
            out = forward_direct + backward_direct
        else:
            out = self.recurrent(time_series, training=training)

        return self.summary_stats(out)