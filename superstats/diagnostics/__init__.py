"""Metrics and plotting helpers for validating superstats workflows."""

from .plots import (
    plot_time_varying_prior,
    plot_time_invariant_prior,
    plot_joint_prior,
    plot_time_varying_posterior,
    plot_marginals,
    plot_pairs,
    plot_forest,
    plot_push_forward,
    plot_time_varying_verification,
    plot_posterior_resimulation,
    plot_recovery,
    plot_calibration,
    plot_z_score_contraction,
)
from .metrics import (
    correlation_per_step,
    nrmse_per_step,
    posterior_contraction_per_step,
    calibration_error_per_step,
)

__all__ = [
    "plot_time_varying_prior",
    "plot_time_invariant_prior",
    "plot_joint_prior",
    "plot_time_varying_posterior",
    "plot_marginals",
    "plot_pairs",
    "plot_forest",
    "plot_push_forward",
    "plot_time_varying_verification",
    "plot_posterior_resimulation",
    "plot_recovery",
    "plot_calibration",
    "plot_z_score_contraction",
    "correlation_per_step",
    "nrmse_per_step",
    "posterior_contraction_per_step",
    "calibration_error_per_step",
]
