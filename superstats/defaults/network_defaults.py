"""Default network constructor arguments for workflows."""

from types import MappingProxyType

DEFAULT_RECURRENT_NETWORK = MappingProxyType(
    {
        "summary_dim": 64,
        "recurrent_type": "gru",
        "hidden_dim": (128, 128),
        "time_embed_dim": 16,
        "time_axis": 0,
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

DEFAULT_COUPLING_FLOW = MappingProxyType({"depth": 6, "transform": "affine"})
DEFAULT_CONSISTENCY_MODEL = MappingProxyType({"subnet_kwargs": {"widths": (256,) * 4}})
