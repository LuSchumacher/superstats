from collections.abc import Sequence
from numbers import Integral, Real

import bayesflow as bf
import keras

from bayesflow.types import Tensor
from bayesflow.utils import layer_kwargs
from bayesflow.utils.serialization import serializable

from .utils import expand_singletons_to_common_length


@serializable("custom")
class RecurrentNet(bf.networks.SummaryNetwork):
    """Implements a sequence-producing recurrent network.

    Parameters
    ----------
    summary_dim    : int, optional, default: 64
        Per-timestep output dimensionality.
    hidden_dim     : int or sequence of int, optional, default: 256
        Dimensionality of the hidden state in each recurrent layer.
    recurrent_type : {"lstm", "gru"} or sequence thereof, optional, default: "lstm"
        Type of recurrent unit to use in each layer.
    bidirectional  : bool or sequence of bool, optional, default: True
        If True, the layer is processed bidirectionally and the two
        directions are summed. If False, the layer processes the
        sequence forward only.
    dropout        : float or sequence of float in [0, 1], optional, default: 0.05
        Dropout rate applied within each recurrent layer.
    **kwargs
        Additional keyword arguments passed to the parent class
        constructor.

    Notes
    -----
    All recurrent layers are built with `return_sequences=True`, so
    the projection is applied per timestep and the sequence-length
    axis is retained. If any of `hidden_dim`, `recurrent_type`,
    `bidirectional`, or `dropout` is a sequence with more than one
    element, all single values are expanded to that length. Multiple
    multi-element sequences must have the same length.

    Raises
    ------
    ValueError
        If per-layer parameter sequences have incompatible lengths or
        contain invalid values.
    """

    def __init__(
        self,
        summary_dim: int = 64,
        hidden_dim: int | Sequence[int] = 256,
        recurrent_type: str | Sequence[str] = "lstm",
        bidirectional: bool | Sequence[bool] = True,
        dropout: float | Sequence[float] = 0.05,
        **kwargs,
    ):
        super().__init__(**layer_kwargs(kwargs))

        recurrent_kwargs = expand_singletons_to_common_length(
            hidden_dim=hidden_dim,
            recurrent_type=recurrent_type,
            bidirectional=bidirectional,
            dropout=dropout
        )

        recurrent_layers = []
        for constructor_kwargs in zip(
            recurrent_kwargs["hidden_dim"],
            recurrent_kwargs["recurrent_type"],
            recurrent_kwargs["bidirectional"],
            recurrent_kwargs["dropout"]
        ):
            hidden_dim_, recurrent_type_, bidirectional_, dropout_ = self._validate_layer_kwargs(*constructor_kwargs)
            recurrent_constructor = self._recurrent_constructor(recurrent_type_)
            recurrent_layer = recurrent_constructor(
                units=hidden_dim_,
                dropout=dropout_,
                return_sequences=True
            )

            if bidirectional_:
                recurrent_layer = keras.layers.Bidirectional(recurrent_layer, merge_mode="sum")

            recurrent_layers.append(recurrent_layer)

        self.recurrent_layers = recurrent_layers

        self.summary_stats = keras.layers.Conv1D(
            filters=summary_dim, 
            kernel_size=1
        )

        self.summary_dim = summary_dim
        self.hidden_dim = hidden_dim
        self.recurrent_type = recurrent_type
        self.bidirectional = bidirectional
        self.dropout = dropout

    def call(self, time_series: Tensor, training: bool = False) -> Tensor:
        """Compute per-timestep summary statistics for a batch of time series.

        Parameters
        ----------
        time_series : Tensor of shape (batch_size, sequence_length, num_features)
            Input time series.
        training    : bool, optional, default: False
            Whether the layer is in training mode (affects dropout).

        Returns
        -------
        summary : Tensor - the learned summary of shape
            (batch_size, sequence_length, summary_dim).
        """
        out = time_series
        for recurrent_layer in self.recurrent_layers:
            out = recurrent_layer(out, training=training)

        return self.summary_stats(out)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "summary_dim": self.summary_dim,
                "hidden_dim": self.hidden_dim,
                "recurrent_type": self.recurrent_type,
                "bidirectional": self.bidirectional,
                "dropout": self.dropout,
            }
        )
        return config

    @staticmethod
    def _recurrent_constructor(recurrent_type: str):
        if recurrent_type == "lstm":
            return keras.layers.LSTM
        if recurrent_type == "gru":
            return keras.layers.GRU

        raise ValueError(f"recurrent_type must be one of ['lstm', 'gru'], not {recurrent_type!r}.")

    @staticmethod
    def _validate_layer_kwargs(
        hidden_dim: int,
        recurrent_type: str,
        bidirectional: bool,
        dropout: float,
    ) -> tuple[int, str, bool, float]:
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, Integral) or hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be a positive integer, not {hidden_dim!r}.")

        if not isinstance(recurrent_type, str):
            raise ValueError(f"recurrent_type must be one of ['lstm', 'gru'], not {recurrent_type!r}.")
        recurrent_type = recurrent_type.lower()

        if not isinstance(bidirectional, bool):
            raise ValueError(f"bidirectional must be a boolean, not {bidirectional!r}.")

        if isinstance(dropout, bool) or not isinstance(dropout, Real) or not 0 <= dropout <= 1:
            raise ValueError(f"dropout must be a float in [0, 1], not {dropout!r}.")

        return int(hidden_dim), recurrent_type, bidirectional, float(dropout)
