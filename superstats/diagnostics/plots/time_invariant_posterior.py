"""Posterior sample visualization helpers."""

from collections.abc import Callable, Mapping, Sequence
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Literal
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

from superstats.defaults import (
    BASE_COLOR,
    BASE_COL_WIDTH,
    BASE_ROW_HEIGHT,
    CATEGORICAL_PALETTE,
    HSPACE,
    LABEL_FONTSIZE,
    LABEL_PAD,
    TICK_FONTSIZE,
    TITLE_FONTSIZE,
    WSPACE,
    Y_LABEL_PAD,
)
from superstats.utils.indexing import format_dataset_label, normalize_data_indices
from superstats.utils.plotting import (
    get_default_num_cols,
    get_layout,
    plot_dist,
    resolve_dist_alpha,
)


def plot_time_invariant_posterior(
    estimates: Mapping[str, np.ndarray] | np.ndarray,
    targets: Mapping[str, np.ndarray] | np.ndarray | None = None,
    variable_keys: Sequence[str] | None = None,
    variable_names: Sequence[str] | None = None,
    aggregation: Callable | None = None,
    mixture_names: dict | None = None,
    dist_type: Literal["hist", "kde", "both"] = "hist",
    num_bins: int | None = None,
    dist_alpha: float | None = None,
    num_cols: int | None = None,
    color: str = BASE_COLOR,
    title_fontsize: int = TITLE_FONTSIZE,
    label_fontsize: int = LABEL_FONTSIZE,
    tick_fontsize: int = TICK_FONTSIZE,
    figsize: tuple[float, float] | None = None,
    data_idx: int | Sequence[int] | None = None,
) -> plt.Figure:
    """Plot time-invariant parameter posteriors.

    Parameters
    ----------
    estimates      : Mapping[str, np.ndarray] or np.ndarray
        Posterior samples. If a dict, values of shape
        (num_datasets, num_post_samples, num_steps, num_components),
        keyed by variable. If an array, shape
        (num_datasets, num_post_samples, num_steps, num_params)
        directly - treated as single-component parameters; mixture
        grouping is not inferable from array input.
    targets        : Mapping[str, np.ndarray], np.ndarray, or None, optional, default: None
        Ground-truth values, matching the input type of `estimates`.
        If a dict, values of shape (num_datasets, num_components). If
        an array, shape (num_datasets, num_params) directly. If
        given, drawn as black dashed vertical lines. When
        `aggregation` is None, one dashed line per panel marks that
        panel's specific dataset's true value. When `aggregation` is
        given, the per-dataset true values are collapsed with
        `aggregation` and a single dashed line is drawn per panel.
    variable_keys  : sequence of str or None, optional, default: None
        Which variables to select and plot, and in what order, when
        `estimates` is a dict. By default, all keys, in dict
        insertion order. Ignored for array input.
    variable_names : sequence of str or None, optional, default: None
        Display names (used for panel labels/titles), in the same
        order as `variable_keys` (or the array's last axis). Defaults
        to `variable_keys` for dict input, or `param_0`, `param_1`,
        ... for array input.
    aggregation    : callable or None, optional, default: None
        Controls both the posterior layout and the target summary.
        If None: one panel per (dataset, parameter) pair; rows=params,
        cols=datasets, param name as row label, dataset index as
        column title; `targets` (if given) are shown per dataset.
        If a callable (e.g. np.mean, np.median): posterior samples are
        pooled across datasets into one panel per parameter, arranged
        in a `num_cols`-column grid; `targets` (if given) are
        collapsed across datasets with `aggregation` into a single
        reference value per panel.
    mixture_names  : dict or None, optional, default: None
        Mapping from base parameter name (e.g. "a", without any
        "_mixture_weights" suffix) to a list of component names.
        Defaults to "component 0", "component 1", ... when not
        supplied. Only applies to dict input with multi-component
        values.
    dist_type      : {"hist", "kde", "both"}, optional, default: "hist"
        Distribution type used for posterior distributions.
    num_bins       : int or None, optional, default: None
        Number of histogram bins. If None, Seaborn selects the bins.
    dist_alpha     : float or None, optional, default: None
        Opacity of posterior distributions. If None, uses 1.0 for a
        single distribution and 0.5 for overlaid mixture components.
    num_cols       : int or None, optional, default: None
        Exact number of grid columns. If None, non-aggregated plots use
        one column per selected dataset and aggregated plots use the
        shared compact dynamic layout.
    color          : str, optional, default: BASE_COLOR
        Base color for non-mixture parameters.
    title_fontsize : int, optional, default: 22
        The font size of the panel titles.
    label_fontsize : int, optional, default: 18
        The font size of the row labels (param names, non-pooled
        layout only).
    tick_fontsize  : int, optional, default: 16
        The font size of the axis tick labels.
    figsize        : tuple of two floats or None, optional, default: None
        Explicit figure size in inches. If None, the default layout size
        is used.
    data_idx       : int, sequence of int, or None, optional, default: None
        Dataset indices to plot. None selects all datasets. A single
        integer preserves the dataset axis, and a sequence preserves
        the requested order.

    Returns
    -------
    fig : plt.Figure - the figure instance for optional saving

    Raises
    ------
    ValueError
        If no variables are found to plot (empty `variable_keys`,
        whether resolved by default or passed explicitly), or if
        `variable_names` doesn't match the number of variables for
        array input, or if a plotting option is invalid.
    """

    mixture_names = mixture_names or {}

    if isinstance(estimates, Mapping):
        keys = list(variable_keys) if variable_keys is not None else list(estimates.keys())
        if not keys:
            raise ValueError("No variables found to plot.")
        missing = [k for k in keys if k not in estimates]
        if missing:
            raise ValueError(f"variable_keys not found in estimates: {missing}")

        names = list(variable_names) if variable_names is not None else keys
        if len(names) != len(keys):
            raise ValueError(f"variable_names has {len(names)} entries but there are {len(keys)} variables.")
        local_estimates = {n: estimates[k] for k, n in zip(keys, names)}
        local_mixture_names = {
            n: mixture_names[k.split("_mixture_weights")[0]]
            for k, n in zip(keys, names)
            if k.split("_mixture_weights")[0] in mixture_names
        }
        local_targets = {n: targets[k] for k, n in zip(keys, names)} if targets is not None else None
    else:
        num_params = estimates.shape[-1]
        names = list(variable_names) if variable_names is not None else [f"param_{p}" for p in range(num_params)]
        if len(names) != num_params:
            raise ValueError(f"variable_names has {len(names)} entries but there are {num_params} variables.")
        if not names:
            raise ValueError("No variables found to plot.")

        local_estimates = {n: estimates[..., p : p + 1] for p, n in enumerate(names)}
        local_mixture_names = {}
        local_targets = {n: targets[..., p : p + 1] for p, n in enumerate(names)} if targets is not None else None

    num_datasets = next(iter(local_estimates.values())).shape[0]
    selected_indices = normalize_data_indices(data_idx, num_datasets)
    local_estimates = {name: np.asarray(values)[selected_indices] for name, values in local_estimates.items()}
    if local_targets is not None:
        local_targets = {name: np.asarray(values)[selected_indices] for name, values in local_targets.items()}
    D = len(selected_indices)

    # panels meta
    panels_meta = []
    for name in names:
        n_components = local_estimates[name].shape[-1]
        if n_components > 1:
            comp_names = local_mixture_names.get(name, [f"component {i}" for i in range(n_components)])
            panels_meta.append((name, list(range(n_components)), comp_names, True))
        else:
            panels_meta.append((name, [0], [name], False))

    P = len(panels_meta)

    # pool across samples and steps
    pooled = {}
    for name in names:
        arr = local_estimates[name]
        B, S, T, C = arr.shape
        for c in range(C):
            pooled[(name, c)] = arr[:, :, :, c].reshape(B, S * T)

    # layout
    if aggregation is None:
        matrix_layout = num_cols is None
        if matrix_layout:
            num_rows = P
            layout_num_cols = D
        else:
            layout_num_cols = num_cols
            num_rows = int(np.ceil(P * D / layout_num_cols))
    else:
        matrix_layout = False
        if num_cols is None:
            num_cols = get_default_num_cols(P)
        num_rows = int(np.ceil(P / num_cols))
        layout_num_cols = num_cols

    plot_figsize, legend_bottom, legend_y = get_layout(
        num_rows,
        layout_num_cols,
        figsize,
        col_width=BASE_COL_WIDTH,
        row_height=BASE_ROW_HEIGHT,
    )
    if aggregation is None and matrix_layout and layout_num_cols == 1 and figsize is None:
        label_font = FontProperties(
            family=plt.rcParams["font.family"],
            size=label_fontsize,
        )
        max_label_width = (
            max(
                TextPath(
                    (0.0, 0.0),
                    name,
                    prop=label_font,
                )
                .get_extents()
                .width
                for name in names
            )
            / 72.0
        )
        plot_figsize = (
            plot_figsize[0] + max_label_width + 0.25,
            plot_figsize[1],
        )

    fig, axes = plt.subplots(
        num_rows,
        layout_num_cols,
        figsize=plot_figsize,
        squeeze=False,
    )

    axes_flat = axes.ravel()

    # per-panel loop
    panel = 0
    legend_handles = []
    legend_labels = set()
    for p, (param_name, comp_indices, comp_labels, is_mixture) in enumerate(panels_meta):
        datasets = range(D) if aggregation is None else [None]
        for d in datasets:
            ax = axes_flat[panel]
            panel_dist_alpha = resolve_dist_alpha(dist_alpha, len(comp_indices))

            for ci, (c, label) in enumerate(zip(comp_indices, comp_labels)):
                c_color = CATEGORICAL_PALETTE[ci % len(CATEGORICAL_PALETTE)] if is_mixture else color
                values = pooled[(param_name, c)][d] if aggregation is None else pooled[(param_name, c)].reshape(-1)
                plot_dist(
                    values,
                    ax=ax,
                    dist_type=dist_type,
                    color=c_color,
                    num_bins=num_bins,
                    alpha=panel_dist_alpha,
                )

                if is_mixture and label not in legend_labels:
                    legend_handles.append(
                        mpatches.Patch(
                            facecolor=c_color,
                            edgecolor="none",
                            alpha=panel_dist_alpha,
                            label=label,
                        )
                    )
                    legend_labels.add(label)

                if local_targets is not None:
                    target_vals = local_targets[param_name][:, c]

                    if aggregation is None:
                        target_val = target_vals[d]
                    else:
                        target_val = aggregation(target_vals, axis=0)

                    ax.axvline(
                        target_val,
                        color="black",
                        linestyle="--",
                        linewidth=1.5,
                        zorder=5,
                    )

            if aggregation is None:
                if matrix_layout:
                    if p == 0:
                        ax.set_title(
                            format_dataset_label(selected_indices[d]),
                            fontsize=title_fontsize,
                            pad=15,
                        )
                    if d == 0:
                        ax.set_ylabel(
                            param_name,
                            fontsize=label_fontsize,
                            rotation=0,
                            ha="right",
                            va="center",
                            labelpad=Y_LABEL_PAD,
                        )
                    else:
                        ax.set_ylabel("")
                    show_xlabel = p == P - 1
                else:
                    title = param_name if D == 1 else f"{param_name} — {format_dataset_label(selected_indices[d])}"
                    ax.set_title(
                        title,
                        fontsize=title_fontsize,
                        pad=15,
                    )
                    ax.set_ylabel(
                        "Density" if panel % layout_num_cols == 0 else "",
                        fontsize=label_fontsize,
                        labelpad=Y_LABEL_PAD,
                    )
                    show_xlabel = panel // layout_num_cols == num_rows - 1
            else:
                ax.set_title(param_name, fontsize=title_fontsize, pad=15)
                ax.set_ylabel(
                    "Density" if panel % layout_num_cols == 0 else "",
                    fontsize=label_fontsize,
                    labelpad=Y_LABEL_PAD,
                )
                show_xlabel = panel // layout_num_cols == num_rows - 1

            ax.set_xlabel(
                "Value" if show_xlabel else "",
                fontsize=label_fontsize,
                labelpad=LABEL_PAD,
            )

            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=tick_fontsize)
            panel += 1

    for j in range(panel, len(axes_flat)):
        axes_flat[j].axis("off")

    if local_targets is not None:
        legend_handles.append(
            mlines.Line2D(
                [],
                [],
                color="black",
                linestyle="--",
                linewidth=1.5,
                label="Target",
            )
        )

    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=min(4, len(legend_handles)),
            fontsize=label_fontsize,
            framealpha=0.0,
            bbox_to_anchor=(0.5, legend_y),
        )

    sns.despine()
    plt.tight_layout()
    fig.subplots_adjust(
        bottom=legend_bottom,
        hspace=HSPACE,
        wspace=WSPACE,
    )
    return fig
