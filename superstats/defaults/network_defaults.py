"""Default network constructor arguments for workflows."""

from types import MappingProxyType

DEFAULT_RECURRENT_NETWORK = MappingProxyType(
    {
        "summary_dim": 64,
        "recurrent_type": "gru",
        "hidden_dim": (128, 128),
        "time_embed_dim": 16,
        "time_axis": 0,
        "return_sequences": True,
    }
)

DEFAULT_TRANSFORMER_NETWORK = MappingProxyType(
    {
        "summary_dim": 64,
        "embed_dims": (128, 128),
        "num_heads": (4, 4),
        "time_embed_dim": 16,
        "time_axis": 0,
        "return_sequences": True,
    }
)

DEFAULT_COUPLING_FLOW = MappingProxyType({"depth": 2, "transform": "spline"})

DEFAULT_AUTOREGRESSIVE_ENCODER_NETWORK = MappingProxyType(
    {
        "summary_dim": 16,
        "embed_dims": (64, 64),
        "num_heads": (4, 4),
        "dropout": 0.05,
        "expansion_factor": 4.0,
        "glu_variant": "swiglu",
        "kernel_initializer": "orthogonal",
        "use_bias": False,
        "layer_norm": True,
        "gate_attention": False,
        "gate_ffn": True,
        "time_embedding": "time2vec",
        "time_embed_dim": 8,
        "time_axis": None,
        "downsample": None,
        "return_sequences": True,
    }
)
DEFAULT_AUTOREGRESSIVE_DECODER_NETWORK = MappingProxyType(
    {
        "embed_dim": 64,
        "output_dim": 64,
        "num_layers": 2,
        "num_heads": 4,
        "dropout": 0.05,
        "expansion_factor": 4.0,
        "glu_variant": "swiglu",
        "use_bias": False,
        "layer_norm": True,
        "include_condition": True,
        "time_embed_dim": 8,
        "kernel_initializer": "glorot_uniform",
    }
)
DEFAULT_FILTERING_ENCODER_NETWORK = MappingProxyType(
    {
        **dict(DEFAULT_RECURRENT_NETWORK),
        "bidirectional": False,
        "return_sequences": True,
    }
)
DEFAULT_FILTERING_DECODER_NETWORK = MappingProxyType(
    {
        "embed_dim": 256,
        "output_dim": 256,
        "recurrent_type": "gru",
        "include_condition": True,
    }
)
DEFAULT_CONSISTENCY_MODEL = MappingProxyType({"subnet_kwargs": {"widths": (256,) * 4}})
