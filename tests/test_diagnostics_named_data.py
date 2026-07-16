import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from superstats.diagnostics.plots import plot_posterior_resimulation, plot_push_forward


def test_plot_push_forward_accepts_named_data():
    rng = np.random.default_rng(0)
    data = {
        "response_time": rng.normal(size=(3, 5)),
        "choice": rng.integers(0, 2, size=(3, 5)).astype(float),
    }

    fig = plot_push_forward(
        data,
        data_dim="response_time",
        kind="trajectory",
        marginal=False,
        uncertainty_fun=None,
    )

    assert fig is not None
    plt.close(fig)


def test_plot_posterior_resimulation_accepts_named_data():
    rng = np.random.default_rng(1)
    pred_data = {
        "response_time": rng.normal(size=(2, 4, 5)),
        "choice": rng.integers(0, 2, size=(2, 4, 5)).astype(float),
    }
    real_data = {
        "response_time": rng.normal(size=(2, 5)),
        "choice": rng.integers(0, 2, size=(2, 5)).astype(float),
    }

    fig = plot_posterior_resimulation(
        pred_data,
        real_data,
        data_dim="response_time",
        kind="trajectory",
        marginal=False,
        uncertainty_fun="std",
    )

    assert fig is not None
    plt.close(fig)
