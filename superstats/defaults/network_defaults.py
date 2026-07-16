"""Default network constructor arguments for workflows."""

from types import MappingProxyType

DEFAULT_RECURRENT_NETWORK = MappingProxyType(
    {
        "recurrent_type": "lstm",
        "hidden_dim": 128,
        "time_axis": 0,
    }
)

DEFAULT_TRANSFORMER_NETWORK = MappingProxyType(
    {
        "summary_dim": 64,
        "time_axis": 0,
        "return_sequences": True,
    }
)

DEFAULT_COUPLING_FLOW = MappingProxyType({"depth": 6, "transform": "affine"})
DEFAULT_CONSISTENCY_MODEL = MappingProxyType({"subnet_kwargs": {"widths": (256,) * 4}})
