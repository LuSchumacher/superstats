"""Recurrent summary network layers."""

from collections.abc import Sequence
from numbers import Integral

import bayesflow as bf

import keras

from bayesflow.types import Tensor
from bayesflow.utils import layer_kwargs
from bayesflow.utils.serialization import serializable
from bayesflow.networks.helpers import Time2Vec

from .utils import expand_singletons_to_common_length


@serializable("custom")
class RecurrentNet(bf.networks.SummaryNetwork):
    """Implements a sequence-producing recurrent network.

    Parameters
    ----------
    summary_dim    : int, optional, default: 64
        Per-timestep output dimensionality.
    hidden_dim     : int or sequence of int, optional, default: (128, 128)
        Dimensionality of the hidden state in each recurrent layer.
    recurrent_type : {"lstm", "gru"} or sequence, optional, default: "lstm"
        Type of recurrent unit to use in each layer.
    bidirectional  : bool or sequence of bool, optional, default: True
        If True, the layer is processed bidirectionally and the two
        directions are merged according to `merge_mode`. If False,
        the layer processes the sequence forward only.
    merge_mode     : {"sum", "mul", "ave", "concat"} or sequence, optional, default: "sum"
        Mode used to merge forward and backward outputs in bidirectional layers.
    layer_norm     : bool or sequence of bool, optional, default: True
        Whether to apply layer normalization after each recurrent layer.
    time_embed_dim : int, optional, default: 16
        The number of features learned by the time2vec preprocessing layer.
    dropout        : float or sequence of float in [0, 1], optional, default: 0.05
        Dropout rate applied after the recurrent layer(s).
    **kwargs
        Additional keyword arguments passed to the parent class
        constructor.

    Notes
    -----
    All recurrent layers are built with `return_sequences=True`, so
    the projection is applied per timestep and the sequence-length
    axis is retained. If any of `hidden_dim`, `recurrent_type`,
    `bidirectional`, `merge_mode`, or `layer_norm` is a sequence with
    more than one element, all single values are expanded to that length. Multiple
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
        hidden_dim: int | Sequence[int] = (128, 128),
        recurrent_type: str | Sequence[str] = "gru",
        bidirectional: bool | Sequence[bool] = True,
        merge_mode: str | Sequence[str] = "sum",
        layer_norm: bool | Sequence[bool] = True,
        time_axis: int = 0,
        time_embed_dim: int = 16,
        dropout: float = 0.05,
        **kwargs,
    ):
        super().__init__(**layer_kwargs(kwargs))

        recurrent_kwargs = expand_singletons_to_common_length(
            hidden_dim=hidden_dim,
            recurrent_type=recurrent_type,
            bidirectional=bidirectional,
            merge_mode=merge_mode,
            layer_norm=layer_norm,
        )

        recurrent_layers = []
        normalization_layers = []
        for constructor_kwargs in zip(
            recurrent_kwargs["hidden_dim"],
            recurrent_kwargs["recurrent_type"],
            recurrent_kwargs["bidirectional"],
            recurrent_kwargs["merge_mode"],
            recurrent_kwargs["layer_norm"],
        ):
            hidden_dim_, recurrent_type_, bidirectional_, merge_mode_, layer_norm_ = self._validate_layer_kwargs(
                *constructor_kwargs
            )
            recurrent_constructor = self._recurrent_constructor(recurrent_type_)
            recurrent_layer = recurrent_constructor(units=hidden_dim_, return_sequences=True)

            if bidirectional_:
                recurrent_layer = keras.layers.Bidirectional(recurrent_layer, merge_mode=merge_mode_)

            recurrent_layers.append(recurrent_layer)
            normalization_layers.append(keras.layers.LayerNormalization(axis=-1) if layer_norm_ else None)

        self.recurrent_layers = recurrent_layers
        self.normalization_layers = normalization_layers

        self.dropout_layer = keras.layers.Dropout(dropout)

        self.summary_stats = keras.layers.Conv1D(filters=summary_dim, kernel_size=1)
        self.time_embedding = Time2Vec(time_embed_dim)

        self.summary_dim = summary_dim
        self.time_axis = time_axis
        self.time_embed_dim = time_embed_dim
        self.hidden_dim = hidden_dim
        self.recurrent_type = recurrent_type
        self.bidirectional = bidirectional
        self.merge_mode = merge_mode
        self.layer_norm = layer_norm
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

        time = time_series[..., self.time_axis]
        indices = list(range(keras.ops.shape(time_series)[-1]))
        indices.pop(self.time_axis)
        out = keras.ops.take(time_series, indices, axis=-1)

        out = self.time_embedding(out, t=time)

        for recurrent_layer, normalization_layer in zip(self.recurrent_layers, self.normalization_layers):
            out = recurrent_layer(out)
            if normalization_layer is not None:
                out = normalization_layer(out, training=training)

        out = self.dropout_layer(out, training=training)

        return self.summary_stats(out)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "summary_dim": self.summary_dim,
                "hidden_dim": self.hidden_dim,
                "recurrent_type": self.recurrent_type,
                "bidirectional": self.bidirectional,
                "merge_mode": self.merge_mode,
                "layer_norm": self.layer_norm,
                "dropout": self.dropout,
                "time_axis": self.time_axis,
                "time_embed_dim": self.time_embed_dim,
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
        merge_mode: str,
        layer_norm: bool,
    ) -> tuple[int, str, bool, str, bool]:
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, Integral) or hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be a positive integer, not {hidden_dim!r}.")

        if not isinstance(recurrent_type, str):
            raise ValueError(f"recurrent_type must be one of ['lstm', 'gru'], not {recurrent_type!r}.")
        recurrent_type = recurrent_type.lower()

        if not isinstance(bidirectional, bool):
            raise ValueError(f"bidirectional must be a boolean, not {bidirectional!r}.")

        if not isinstance(merge_mode, str):
            raise ValueError(f"merge_mode must be one of ['sum', 'mul', 'ave', 'concat'], not {merge_mode!r}.")
        merge_mode = merge_mode.lower()

        if merge_mode not in {"sum", "mul", "ave", "concat"}:
            raise ValueError(f"merge_mode must be one of ['sum', 'mul', 'ave', 'concat'], not {merge_mode!r}.")

        if not isinstance(layer_norm, bool):
            raise ValueError(f"layer_norm must be a boolean, not {layer_norm!r}.")

        return hidden_dim, recurrent_type, bidirectional, merge_mode, layer_norm
