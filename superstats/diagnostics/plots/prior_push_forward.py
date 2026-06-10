import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_push_forward(
    data: np.ndarray,
    data_dim: int = 0,
    type: str = "hist",
    aggregate: bool = False,
    stats_fun: callable = None,
    n_cols: int = 3,
    color: str = "#822621",
    title_fontsize: int = 14,
    tick_fontsize: int = 10,
    alpha: float = 0.8,
):
    """
    Plot prior push-forward for a single data dimension.

    Parameters
    ----------
    data : np.ndarray, shape (batch_size, steps, data_dims)
    data_dim : int
        Which data dimension to plot.
    type : "hist" | "trajectory"
    aggregate : bool
        If True, aggregate across batch before plotting.
    stats_fun : callable, optional
        Applied per batch item when type="hist" and aggregate=True.
        Default is mean across steps.
    """
    x = data[:, :, data_dim]
    batch_size, steps = x.shape

    COL_WIDTH, ROW_HEIGHT = 4.0, 3.0

    if type == "trajectory" and aggregate:
        mean  = x.mean(axis=0)
        lower = np.percentile(x, 2.5,  axis=0)
        upper = np.percentile(x, 97.5, axis=0)
        t = np.arange(steps)

        fig, ax = plt.subplots(figsize=(COL_WIDTH * 2, ROW_HEIGHT))
        ax.plot(t, mean, color=color, linewidth=2.0)
        ax.fill_between(t, lower, upper, color=color, alpha=0.25)
        ax.set_xlabel("Step", fontsize=tick_fontsize)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=tick_fontsize)

    elif type == "hist" and aggregate:
        fn    = stats_fun if stats_fun is not None else lambda x: x.mean(axis=-1)
        stats = np.asarray(fn(x)).reshape(-1)

        fig, ax = plt.subplots(figsize=(COL_WIDTH * 2, ROW_HEIGHT))
        sns.histplot(
            stats,
            bins=30, stat="density", kde=True,
            line_kws={"linewidth": 2.0},
            ax=ax, color=color, alpha=alpha,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=tick_fontsize)

    elif type == "trajectory":
        n_rows = int(np.ceil(batch_size / n_cols))
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(COL_WIDTH * n_cols, ROW_HEIGHT * n_rows),
        )
        axes = np.atleast_1d(axes).ravel()
        t = np.arange(steps)

        for i in range(batch_size):
            ax = axes[i]
            ax.plot(t, x[i], color=color, alpha=alpha, linewidth=1.5)
            ax.set_title(f"Dataset {i}", fontsize=title_fontsize)
            ax.set_xlabel("Step", fontsize=tick_fontsize)
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=tick_fontsize)

        for j in range(batch_size, len(axes)):
            axes[j].axis("off")

    else:
        n_rows = int(np.ceil(batch_size / n_cols))
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(COL_WIDTH * n_cols, ROW_HEIGHT * n_rows),
        )
        axes = np.atleast_1d(axes).ravel()

        for i in range(batch_size):
            ax = axes[i]
            sns.histplot(
                x[i],
                bins=30, stat="density", kde=True,
                line_kws={"linewidth": 2.0},
                ax=ax, color=color, alpha=alpha,
            )
            ax.set_title(f"Dataset {i}", fontsize=title_fontsize)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=tick_fontsize)

        for j in range(batch_size, len(axes)):
            axes[j].axis("off")

    sns.despine()
    plt.tight_layout()

    return fig