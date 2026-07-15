"""Default configuration values used across superstats."""

from .network_defaults import (
    DEFAULT_SUMMARY_NETWORK,
    DEFAULT_INFERENCE_NETWORK,
)

from .color_palette import BASE_COLOR, METRIC_COLORS, CATEGORICAL_PALETTE

__all__ = [
    "DEFAULT_SUMMARY_NETWORK",
    "DEFAULT_INFERENCE_NETWORK",
    "BASE_COLOR",
    "METRIC_COLORS",
    "CATEGORICAL_PALETTE",
]
