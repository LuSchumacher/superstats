"""Plotting functions for priors, posteriors, and recovery diagnostics."""

import matplotlib.pyplot as plt
from matplotlib import font_manager

from .calibration import plot_calibration
from .joint_prior import plot_joint_prior
from .recovery import plot_recovery
from .time_invariant_posterior import plot_time_invariant_posterior
from .time_invariant_prior import plot_time_invariant_prior
from .time_varying_posterior import plot_time_varying_posterior
from .time_varying_prior import plot_time_varying_prior
from .prior_push_forward import plot_push_forward
from .time_varying_verification import plot_time_varying_verification
from .posterior_resimulation import plot_posterior_resimulation
from .z_score_contraction import plot_z_score_contraction

plt.rcParams["axes.axisbelow"] = True
_font_family = "Inter" if any(font.name == "Inter" for font in font_manager.fontManager.ttflist) else "DejaVu Sans"
plt.rcParams["font.family"] = _font_family
plt.rcParams["mathtext.fontset"] = "cm"  # Computer Modern

__all__ = [
    "plot_time_varying_prior",
    "plot_time_invariant_prior",
    "plot_joint_prior",
    "plot_time_varying_posterior",
    "plot_time_invariant_posterior",
    "plot_push_forward",
    "plot_time_varying_verification",
    "plot_posterior_resimulation",
    "plot_recovery",
    "plot_calibration",
    "plot_z_score_contraction",
]
