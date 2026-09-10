"""Time-invariant posterior z-score and contraction plot."""

from collections.abc import Mapping, Sequence

import bayesflow as bf
import numpy as np

from superstats.defaults import (
    BASE_COLOR,
    LABEL_FONTSIZE,
    TICK_FONTSIZE,
    TITLE_FONTSIZE,
)
from superstats.utils import prepare_plot_data


def plot_z_score_contraction(
    estimates: Mapping[str, np.ndarray] | np.ndarray,
    targets: Mapping[str, np.ndarray] | np.ndarray,
    variable_keys: Sequence[str] | None = None,
    variable_names: Sequence[str] | None = None,
    color: str = BASE_COLOR,
    title_fontsize: int = TITLE_FONTSIZE,
    label_fontsize: int = LABEL_FONTSIZE,
    tick_fontsize: int = TICK_FONTSIZE,
    **kwargs,
):
    """Plot posterior z-scores against posterior contraction.

    This thin wrapper around
    `bf.diagnostics.plots.z_score_contraction` accepts the same
    dict-or-array inputs as the other time-invariant diagnostics.

    Parameters
    ----------
    estimates      : Mapping[str, np.ndarray] or np.ndarray
        Posterior draws. Array input has shape
        (num_sims, num_samples, num_params).
    targets        : Mapping[str, np.ndarray] or np.ndarray
        Ground-truth parameters. Array input has shape
        (num_sims, num_params).
    variable_keys  : sequence of str or None, optional, default: None
        Variables to select from mapping inputs.
    variable_names : sequence of str or None, optional, default: None
        Display names for the selected variables.
    color          : str, optional, default: BASE_COLOR
        Color for the plotted points.
    title_fontsize : int, optional, default: 22
        Font size for panel titles.
    label_fontsize : int, optional, default: 18
        Font size for axis labels.
    tick_fontsize  : int, optional, default: 16
        Font size for tick labels.
    **kwargs
        Forwarded to `bf.diagnostics.plots.z_score_contraction`, such
        as `figsize`, `num_row`, `num_col`, or `markersize`.

    Returns
    -------
    fig : plt.Figure
        The z-score and contraction diagnostic figure.
    """
    estimates_arr, targets_arr, names = prepare_plot_data(
        estimates,
        targets,
        variable_keys,
        variable_names,
    )

    return bf.diagnostics.plots.z_score_contraction(
        estimates=estimates_arr,
        targets=targets_arr,
        variable_names=names,
        color=color,
        title_fontsize=title_fontsize,
        label_fontsize=label_fontsize,
        tick_fontsize=tick_fontsize,
        **kwargs,
    )
