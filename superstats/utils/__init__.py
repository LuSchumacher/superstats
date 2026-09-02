"""General utility functions used across superstats."""

from .transformations import scaled_sigmoid
from .plotting import (
    UNCERTAINTY_BAND_LABELS,
    compute_uncertainty_band,
    compute_uncertainty_bands,
    get_uncertainty_band_label,
    get_layout,
    plot_dist,
    plot_uncertainty_band,
    plot_uncertainty_bands,
    prepare_plot_data,
    resolve_dist_alpha,
    smooth_trajectories,
)
from .dispatch import (
    find_contamination,
    find_inference_network,
    find_missing,
    find_embedding_network,
)
from .logging import error, info, logger, warn_once, warning

__all__ = [
    "scaled_sigmoid",
    "UNCERTAINTY_BAND_LABELS",
    "compute_uncertainty_band",
    "compute_uncertainty_bands",
    "get_uncertainty_band_label",
    "prepare_plot_data",
    "get_layout",
    "plot_dist",
    "plot_uncertainty_band",
    "plot_uncertainty_bands",
    "resolve_dist_alpha",
    "smooth_trajectories",
    "find_inference_network",
    "find_missing",
    "find_contamination",
    "find_embedding_network",
    "logger",
    "info",
    "warning",
    "error",
    "warn_once",
]
