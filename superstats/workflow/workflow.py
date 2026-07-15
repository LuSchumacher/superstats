"""High-level workflow wrapper around BayesFlow."""

from typing import Literal, Callable
from collections.abc import Mapping, Sequence

import bayesflow as bf
import numpy as np
import keras
import functools
import os
import pickle

from bayesflow.adapters import Adapter
from bayesflow.networks import SummaryNetwork, InferenceNetwork

from superstats.simulation import GenerativeModel
from superstats.networks import RecurrentNet
from superstats.defaults.network_defaults import (
    DEFAULT_SUMMARY_NETWORK,
    DEFAULT_INFERENCE_NETWORK,
)
from superstats.diagnostics.plots import (
    plot_time_varying_verification,
    plot_recovery,
    plot_calibration,
    plot_time_varying_posterior,
    plot_time_invariant_posterior,
)


class Workflow:
    """Lightweight amortized Bayesian inference workflow wrapper.

    Wraps `bf.BasicWorkflow` with sensible defaults for the summary and
    inference networks, an auto-built adapter when one isn't supplied,
    and optional checkpoint/history restoration.

    Parameters
    ----------
    simulator            : GenerativeModel or None, optional, default: None
        The simulator used for training and, when `adapter` is not
        provided, for building a default adapter. Required in that case.
    adapter              : Adapter or None, optional, default: None
        Data adapter for the workflow. If None, a default adapter is
        built from `simulator.local_keys`, `simulator.hyper_keys`, and
        `simulator.shared_keys` (which requires `simulator` to be set).
    summary_network      : {"recurrent"} or SummaryNetwork or None, optional, default: "recurrent"
        "recurrent" builds a `RecurrentNet` using
        `DEFAULT_SUMMARY_NETWORK`; otherwise, the given network (or
        None) is used directly.
    inference_network    : {"consistency"} or InferenceNetwork, optional, default: "consistency"
        "consistency" builds a `bf.networks.StableConsistencyModel`
        using `DEFAULT_INFERENCE_NETWORK`; otherwise, the given network
        is used directly.
    checkpoint_filepath  : str or None, optional, default: None
        Directory for saving/restoring the approximator and training
        history.
    restore_approximator : bool, optional, default: True
        If True and a checkpoint directory exists at
        `checkpoint_filepath`, restore the approximator from it.
        Otherwise, a warning is issued and training starts from scratch.
    restore_history      : bool, optional, default: True
        If True and a `history.pkl` file exists at
        `checkpoint_filepath`, restore `self.history` from it.
        Otherwise, a warning is issued.
    **kwargs
        Forwarded to `bf.BasicWorkflow`.
    """

    def __init__(
        self,
        simulator: GenerativeModel | None = None,
        adapter: Adapter | None = None,
        summary_network: Literal["recurrent"] | SummaryNetwork | None = "recurrent",
        inference_network: Literal["consistency"] | InferenceNetwork = "consistency",
        checkpoint_filepath: str | None = None,
        restore_approximator: bool = True,
        restore_history: bool = True,
        **kwargs,
    ):
        self.simulator = simulator

        if summary_network == "recurrent":
            self.summary_network = RecurrentNet(**DEFAULT_SUMMARY_NETWORK)
        else:
            self.summary_network = summary_network

        if inference_network == "consistency":
            self.inference_network = bf.networks.CouplingFlow(**DEFAULT_INFERENCE_NETWORK)
        else:
            self.inference_network = inference_network

        if adapter is not None:
            self.adapter = adapter
        else:
            self.local_keys = self.simulator.local_keys
            self.hyper_keys = self.simulator.hyper_keys
            self.shared_keys = self.simulator.shared_keys

            adapter = (
                bf.Adapter()
                .convert_dtype("float64", "float32")
                .concatenate(self.local_keys + self.hyper_keys + self.shared_keys, into="inference_variables")
                .as_time_series("time_steps")
            )

            if hasattr(self.simulator, "has_mask") and self.simulator.has_mask:
                adapter = adapter.as_time_series("missing_mask")
                self.adapter = adapter.concatenate(["time_steps", "data", "missing_mask"], into="summary_variables")
            else:
                self.adapter = adapter.concatenate(["time_steps", "data"], into="summary_variables")

        self.checkpoint_filepath = checkpoint_filepath

        if restore_approximator and self.checkpoint_filepath is not None and os.path.isdir(self.checkpoint_filepath):
            restore = True
        elif restore_approximator:
            restore = False
        else:
            restore = False

        self.workflow = bf.BasicWorkflow(
            simulator=self.simulator,
            adapter=self.adapter,
            summary_network=self.summary_network,
            inference_network=self.inference_network,
            standardize="all",
            checkpoint_filepath=self.checkpoint_filepath,
            restore=restore,
            **kwargs,
        )

        if restore_history and self.checkpoint_filepath is not None and os.path.isdir(self.checkpoint_filepath):
            self._load_history()

    def _load_history(self) -> None:
        """Load persisted training history from `checkpoint_filepath`, if present.

        A no-op if `checkpoint_filepath` is None or no `history.pkl`
        file exists there.
        """
        if self.checkpoint_filepath is None:
            return
        path = os.path.join(self.checkpoint_filepath, "history.pkl")
        if not os.path.exists(path):
            return
        with open(path, "rb") as f:
            self.workflow.history = pickle.load(f)

    def _save_history(self, new_history: keras.callbacks.History) -> None:
        """Merge and persist training history to `checkpoint_filepath`, if present.

        A no-op if `checkpoint_filepath` is None.

        Parameters
        ----------
        new_history : keras.callbacks.History
            History from the most recent training run. Merged into any
            existing `self.workflow.history` before saving.
        """
        if self.checkpoint_filepath is None:
            return
        existing = self.workflow.history
        if existing is not None and existing is not new_history:
            for key, values in new_history.history.items():
                existing.history.setdefault(key, []).extend(values)
            new_history = existing
        os.makedirs(self.checkpoint_filepath, exist_ok=True)
        with open(os.path.join(self.checkpoint_filepath, "history.pkl"), "wb") as f:
            pickle.dump(new_history, f)
        self.workflow.history = new_history

    def fit_offline(
        self, data, validation_data, epochs: int = 100, batch_size: int = 32, save_history: bool = True, **kwargs
    ) -> keras.callbacks.History:
        """Train the approximator on a fixed, pre-simulated dataset.

        Parameters
        ----------
        data            : Any
            Training data, in the format expected by
            `bf.BasicWorkflow.fit_offline`.
        validation_data : Any
            Validation data, in the same format as `data`.
        epochs          : int, optional, default: 100
            Number of training epochs.
        batch_size      : int, optional, default: 32
            Training batch size.
        save_history    : bool, optional, default: True
            If True, merge this run's history into `self.history` and
            persist it to `checkpoint_filepath` (if set).
        **kwargs
            Forwarded to `bf.BasicWorkflow.fit_offline`.

        Returns
        -------
        history : keras.callbacks.History - the training history for
            this run
        """
        history = self.workflow.fit_offline(
            data=data, epochs=epochs, batch_size=batch_size, validation_data=validation_data, **kwargs
        )

        if save_history:
            self._save_history(history)

        return history

    def fit_online(
        self,
        num_steps: int,
        epochs: int = 100,
        num_batches_per_epoch: int = 100,
        batch_size: int = 32,
        save_history: bool = True,
        **kwargs,
    ) -> keras.callbacks.History:
        """Train the approximator by simulating data on the fly.

        Temporarily binds `self.simulator.sample` to always draw
        trajectories of length `num_steps` with `tile_to_steps=True`,
        then restores the original method afterward (even if training
        raises).

        Parameters
        ----------
        num_steps             : int
            Number of time steps per simulated trajectory during
            training.
        epochs                : int, optional, default: 100
            Number of training epochs.
        num_batches_per_epoch : int, optional, default: 100
            Number of simulated batches per epoch.
        batch_size            : int, optional, default: 32
            Training batch size.
        save_history          : bool, optional, default: True
            If True, merge this run's history into `self.history` and
            persist it to `checkpoint_filepath` (if set).
        **kwargs
            Forwarded to `bf.BasicWorkflow.fit_online`.

        Returns
        -------
        history : keras.callbacks.History - the training history for
            this run
        """
        original_sample = self.simulator.sample
        self.simulator.sample = functools.partial(original_sample, num_steps=num_steps, tile_to_steps=True)
        try:
            history = self.workflow.fit_online(
                epochs=epochs, num_batches_per_epoch=num_batches_per_epoch, batch_size=batch_size, **kwargs
            )
        finally:
            self.simulator.sample = original_sample

        if save_history:
            self._save_history(history)

        return history

    @property
    def history(self):
        """keras.callbacks.History or None - the workflow's training history."""
        return self.workflow.history

    @property
    def approximator(self):
        """The underlying trained bf approximator object."""
        return self.workflow.approximator

    def sample(
        self,
        data: dict[str, np.ndarray],
        num_samples: int = 500,
        batch_size: int = 4,
        **kwargs,
    ) -> dict[str, np.ndarray]:
        """Run inference on observed data.

        Parameters
        ----------
        data                 : np.ndarray of shape (num_datasets, num_steps, data_dims)
            Observed data to condition on.
        num_samples          : int, optional, default: 500
            Number of posterior samples per dataset.
        batch_size : int, optional, default: 4
            Datasets per GPU batch, to avoid out-of-memory errors.
        **kwargs
            Forwarded to `self.approximator.sample`.

        Returns
        -------
        samples : dict of {param_name: np.ndarray} - posterior samples
            per parameter
        """
        samples = self.approximator.sample(conditions=data, num_samples=num_samples, batch_size=batch_size, **kwargs)
        return samples

    def resimulate_posterior(
        self,
        posterior_samples: Mapping[str, np.ndarray],
        num_sims: int = 10,
        rng=None,
    ) -> np.ndarray:
        """Generate posterior predictive simulations from posterior parameter draws.

        Parameters
        ----------
        posterior_samples : dict of np.ndarray
            Posterior samples returned by `self.sample`. Each array
            should have shape (batch_size, num_samples, num_steps, dim)
            or (batch_size, num_samples, num_steps).
        num_sims          : int, optional, default: 10
            Number of posterior predictive trajectories to simulate
            per dataset.
        rng               : int or np.random.Generator or None, optional, default: None
            Random seed or generator for sampling posterior indices.

        Returns
        -------
        sim_data : np.ndarray of shape (batch_size, num_sims,
            num_steps, data_dim) - simulated data

        Raises
        ------
        ValueError
            If `posterior_samples` is empty, if a posterior array has
            fewer than 3 dimensions, if a parameter's batch size
            doesn't match the others, if a parameter has an unsupported
            number of dimensions, or if a parameter's shape can't be
            reshaped to collapse the sample axis into the batch axis.
        """
        rng = np.random.default_rng(rng)

        if not posterior_samples:
            raise ValueError("posterior_samples must be a non-empty dict.")

        # Infer shape from one posterior parameter
        example = next(iter(posterior_samples.values()))
        if example.ndim < 3:
            raise ValueError(
                "Posterior sample arrays must have at least 3 dimensions: (batch_size, num_samples, num_steps, ...)."
            )

        batch_size, num_draws = example.shape[:2]
        num_steps = example.shape[2]

        sample_idx = rng.integers(num_draws, size=(batch_size, num_sims))

        simulation_params: dict[str, np.ndarray] = {}
        fixed_params = self.simulator.get_fixed_params()

        for name, arr in posterior_samples.items():
            arr = np.asarray(arr)
            if arr.shape[0] != batch_size:
                raise ValueError(
                    f"Posterior parameter '{name}' has batch size {arr.shape[0]} but expected {batch_size}."
                )

            if arr.ndim == 3:
                selected = arr[np.arange(batch_size)[:, None], sample_idx, :]
                simulation_params[name] = selected
            elif arr.ndim == 4:
                selected = arr[
                    np.arange(batch_size)[:, None, None],
                    sample_idx[:, :, None],
                    np.arange(num_steps)[None, None, :],
                    :,
                ]
                simulation_params[name] = selected
            else:
                raise ValueError(
                    f"Unexpected posterior shape for '{name}': {arr.shape}. "
                    "Expected (batch, samples, steps, dim) or (batch, samples, steps)."
                )

        # Collapse sample axis into batch axis for simulation
        expanded_params: dict[str, np.ndarray] = {}
        for name, arr in simulation_params.items():
            if arr.ndim == 2:
                expanded_params[name] = arr.reshape(batch_size * num_sims, num_steps)
            elif arr.ndim == 3:
                expanded_params[name] = arr.reshape(batch_size * num_sims, num_steps, arr.shape[2])
            elif arr.ndim == 4:
                expanded_params[name] = arr.reshape(batch_size * num_sims, num_steps, arr.shape[3])
            else:
                raise ValueError(f"Cannot reshape posterior parameter '{name}' with shape {arr.shape}.")

        for name, value in fixed_params.items():
            expanded_params[name] = np.broadcast_to(np.asarray(value), (batch_size * num_sims,))

        raw_sim = self.simulator.simulate_from_parameters(
            expanded_params,
            batch_size=batch_size * num_sims,
            num_steps=num_steps,
        )

        return raw_sim.reshape(batch_size, num_sims, num_steps, *raw_sim.shape[2:])

    def plot_history(self, history):
        """Plot training loss curves.

        Parameters
        ----------
        history : keras.callbacks.History
            Training history, e.g. from `fit_offline`, `fit_online`, or
            `self.history`.

        Returns
        -------
        fig : plt.Figure - the loss curve figure
        """
        return bf.diagnostics.plots.loss(history, train_color="#822621")

    def verify_time_varying(
        self,
        targets: dict,
        estimates: dict,
        variable_keys: list | None = None,
        variable_names: list | None = None,
        aggregation: Callable = np.median,
        **kwargs,
    ):
        """Plot recovery diagnostics over steps for time-varying parameters.

        Parameters
        ----------
        targets        : dict
            Ground-truth local parameter trajectories, keyed by
            parameter name; each value has shape
            (batch_size, num_steps, 1).
        estimates      : dict
            Posterior estimates for the same parameters, keyed by name;
            each value has shape
            (batch_size, num_post_samples, num_steps, 1).
        variable_keys  : list of str or None, optional, default: None
            Which parameters to select and plot, and in what order.
            Defaults to `self.simulator.local_keys` when not supplied.
        variable_names : list of str or None, optional, default: None
            Display names for the plotted columns, in the same order as
            `variable_keys`. Defaults to `variable_keys` when not
            supplied.
        aggregation    : callable, optional, default: np.median
            Aggregation function forwarded to
            `plot_time_varying_verification`, used to collapse each
            metric across simulations. Typically np.mean or np.median.
        **kwargs
            Additional keyword arguments forwarded to
            `plot_time_varying_verification` (e.g. `colors`,
            `title_fontsize`).

        Returns
        -------
        fig : plt.Figure - the figure instance for optional saving
        """
        local_keys = self.simulator.local_keys

        if variable_keys is None:
            variable_keys = local_keys

        targets_squeezed = {k: targets[k][..., 0] for k in variable_keys}
        estimates_squeezed = {k: estimates[k][..., 0] for k in variable_keys}

        return plot_time_varying_verification(
            estimates=estimates_squeezed,
            targets=targets_squeezed,
            variable_keys=variable_keys,
            variable_names=variable_names,
            aggregation=aggregation,
            **kwargs,
        )

    def verify_time_invariant(
        self,
        targets: Mapping[str, np.ndarray] | np.ndarray,
        estimates: Mapping[str, np.ndarray] | np.ndarray,
        variable_keys: Sequence[str] | None = None,
        variable_names: Sequence[str] | None = None,
        **kwargs,
    ):
        """Plot time-invariant parameter recovery and calibration.

        Parameters
        ----------
        targets        : Mapping[str, np.ndarray] or np.ndarray
            If a dict, mapping from parameter name to an np.ndarray of
            shape (num_sims, dim). If an array, the fully-prepared target
            array of shape (num_sims, num_params) directly (e.g. mixture
            components already expanded).
        estimates      : Mapping[str, np.ndarray] or np.ndarray
            If a dict, mapping from parameter name to an np.ndarray of
            shape (num_sims, num_samples, steps, dim). If an array, the
            fully-prepared estimate array of shape
            (num_sims, num_pooled_samples, num_params) directly. Must use
            the same input type (dict or array) as `targets`.
        variable_keys  : sequence of str or None, optional, default: None
            Which time-invariant parameters to include, and in what
            order, when `targets`/`estimates` are dicts. Defaults to
            `self.simulator.hyper_keys + self.simulator.shared_keys` when
            not supplied. Mixture parameters (dim > 1) are expanded into
            one column per component regardless of this selection.
            Ignored for array input.
        variable_names : sequence of str or None, optional, default: None
            Display names for the final, expanded columns. For dict
            input, must match the number of expanded columns (not
            `len(variable_keys)`) and defaults to the auto-derived
            per-component names. For array input, defaults to `param_0`,
            `param_1`, ...
        **kwargs
            Forwarded to both `plot_recovery` and `plot_calibration` (e.g.
            `label_fontsize`, `title_fontsize`, `tick_fontsize`). Note
            `plot_recovery` takes `color` while `plot_calibration` takes
            `rank_ecdf_color` - pass whichever applies, or both, via
            `**kwargs`.

        Returns
        -------
        figs : tuple - `(fig_recovery, fig_calibration)`, the recovery
            and calibration diagnostic figures

        Raises
        ------
        ValueError
            If no time-invariant parameters are found for dict input.
        """
        if not isinstance(estimates, Mapping):
            fig_recovery = plot_recovery(
                estimates=estimates,
                targets=targets,
                variable_keys=variable_keys,
                variable_names=variable_names,
                **kwargs,
            )
            fig_calibration = plot_calibration(
                estimates=estimates,
                targets=targets,
                variable_keys=variable_keys,
                variable_names=variable_names,
                **kwargs,
            )
            return fig_recovery, fig_calibration

        if variable_keys is None:
            variable_keys = self.simulator.hyper_keys + self.simulator.shared_keys
        if not variable_keys:
            raise ValueError("No time-invariant parameters found.")
        missing = [k for k in variable_keys if k not in estimates or k not in targets]
        if missing:
            raise ValueError(f"variable_keys not found in both estimates and targets: {missing}")

        target_list = []
        estimate_list = []
        expanded_names = []

        for k in variable_keys:
            t_arr = targets[k]
            e_arr = estimates[k]
            B, S, T, dim = e_arr.shape

            e_agg = e_arr.reshape(B, S * T, dim)

            if dim > 1:
                param_key = k.split("_mixture_weights")[0]
                mixture_obj = self.simulator.prior.params.get(param_key)
                if hasattr(mixture_obj, "names") and len(mixture_obj.names) == dim:
                    comp_names = [f"{k}_{n}" for n in mixture_obj.names]
                else:
                    comp_names = [f"{k}_{i}" for i in range(dim)]

                for i, name in enumerate(comp_names):
                    target_list.append(t_arr[:, i : i + 1])
                    estimate_list.append(e_agg[:, :, i : i + 1])
                    expanded_names.append(name)
            else:
                target_list.append(t_arr)
                estimate_list.append(e_agg)
                expanded_names.append(k)

        target_arr = np.concatenate(target_list, axis=-1)
        estimate_arr = np.concatenate(estimate_list, axis=-1)

        if variable_names is not None:
            if len(variable_names) != len(expanded_names):
                raise ValueError(
                    f"variable_names has {len(variable_names)} entries but there are "
                    f"{len(expanded_names)} expanded columns."
                )
            expanded_names = list(variable_names)

        fig_recovery = plot_recovery(
            estimates=estimate_arr,
            targets=target_arr,
            variable_names=expanded_names,
            **kwargs,
        )

        fig_calibration = plot_calibration(
            estimates=estimate_arr,
            targets=target_arr,
            variable_names=expanded_names,
            **kwargs,
        )

        return fig_recovery, fig_calibration

    def plot_time_varying_posterior(
        self,
        estimates: Mapping[str, np.ndarray] | np.ndarray,
        targets: Mapping[str, np.ndarray] | np.ndarray | None = None,
        variable_keys: Sequence[str] | None = None,
        variable_names: Sequence[str] | None = None,
        aggregation: Callable | None = None,
        aggregate_strategy: Literal["full_uncertainty", "no_epistemic"] = "full_uncertainty",
        uncertainty_fun: Literal["std", "95ci", "mad", "95hdi"] | Callable | None = "95ci",
        smoothing: Literal["sma", "ema"] | None = None,
        smoothing_window: int = 5,
        marginal: bool = True,
        **kwargs,
    ):
        """Plot time-varying posterior diagnostics.

        Parameters
        ----------
        estimates          : Mapping[str, np.ndarray] or np.ndarray
            Posterior samples. If a dict, values of shape
            (num_datasets, num_post_samples, num_steps, 1), keyed by
            variable. If an array, shape
            (num_datasets, num_post_samples, num_steps, num_params)
            directly.
        targets            : Mapping[str, np.ndarray], np.ndarray, or None, optional, default: None
            Ground-truth trajectories, matching the input type of
            `estimates`. If a dict, values of shape
            (num_datasets, num_steps, 1). If an array, shape
            (num_datasets, num_steps, num_params) directly. If given,
            drawn as a black dashed line on top of each panel: the raw
            per-dataset trajectory when `aggregation` is None, or
            aggregated across datasets (using `aggregation`) when
            `aggregation` is not None.
        variable_keys      : sequence of str or None, optional, default: None
            Which variables to select and plot, and in what order, when
            `estimates`/`targets` are dicts. Defaults to
            `self.simulator.local_keys` when not supplied. Ignored for
            array input.
        variable_names     : sequence of str or None, optional, default: None
            Display names (used for panel labels/titles), in the same
            order as `variable_keys` (or the array's last axis). Defaults
            to `variable_keys` for dict input, or `param_0`, `param_1`,
            ... for array input.
        aggregation        : callable or None, optional, default: None
            None: one panel per (param, dataset).
            callable: one panel per param, aggregated across datasets.
            Called as `aggregation(trajectories, axis=0)` and must return
            a (T,) center. The same function aggregates `targets` across
            datasets when both `targets` and `aggregation` are given.
        aggregate_strategy : {"full_uncertainty", "no_epistemic"}, optional, default: "full_uncertainty"
            Only used when `aggregation` is not None.
            "full_uncertainty": flatten datasets and posterior samples,
            then summarize.
            "no_epistemic": median across posterior samples per dataset
            first, then aggregate.
        uncertainty_fun    : {"std", "95ci", "mad", "95hdi"} or callable or None, optional, default: "95ci"
            Band drawn around the center line. A callable receives (N, T)
            trajectories and must return `(lo, hi)`, each of shape (T,).
        smoothing          : {"sma", "ema"} or None, optional, default: None
            Applied to each trajectory before computing the center,
            uncertainty, and marginal.
        smoothing_window   : int, optional, default: 5
            Window size for `sma`, or span parameter for `ema`.
        marginal           : bool, optional, default: True
            Attach a marginal KDE panel to the right of each trajectory
            axis. The KDE is computed on the same array used for the
            uncertainty band.
        **kwargs
            Forwarded to `plot_time_varying_posterior` (e.g. `num_cols`,
            `color`, `alpha`, `title_fontsize`, `label_fontsize`,
            `tick_fontsize`, `figsize`).

        Returns
        -------
        fig : plt.Figure - the figure instance for optional saving
        """
        if variable_keys is None:
            variable_keys = self.simulator.local_keys

        return plot_time_varying_posterior(
            estimates=estimates,
            targets=targets,
            variable_keys=variable_keys,
            variable_names=variable_names,
            aggregation=aggregation,
            aggregate_strategy=aggregate_strategy,
            uncertainty_fun=uncertainty_fun,
            smoothing=smoothing,
            smoothing_window=smoothing_window,
            marginal=marginal,
            **kwargs,
        )

    def plot_time_invariant_posterior(
        self,
        estimates: Mapping[str, np.ndarray] | np.ndarray,
        targets: Mapping[str, np.ndarray] | np.ndarray | None = None,
        variable_keys: Sequence[str] | None = None,
        variable_names: Sequence[str] | None = None,
        aggregation: Callable | None = None,
        mixture_names: dict | None = None,
        **kwargs,
    ):
        """Plot time-invariant posterior diagnostics.

        Parameters
        ----------
        estimates      : Mapping[str, np.ndarray] or np.ndarray
            Posterior samples. If a dict, values of shape
            (num_datasets, num_post_samples, num_steps, num_components),
            keyed by variable. If an array, shape
            (num_datasets, num_post_samples, num_steps, num_params)
            directly.
        targets        : Mapping[str, np.ndarray], np.ndarray, or None, optional, default: None
            Ground-truth values, matching the input type of `estimates`.
            If given, drawn as black dashed vertical lines - per dataset
            when `aggregation` is None, or collapsed with `aggregation`
            into a single line per panel otherwise.
        variable_keys  : sequence of str or None, optional, default: None
            Which variables to select and plot, and in what order, when
            `estimates` is a dict. Defaults to
            `self.simulator.hyper_keys + self.simulator.shared_keys` when
            not supplied. Ignored for array input.
        variable_names : sequence of str or None, optional, default: None
            Display names for the plotted panels. Defaults to
            `variable_keys` (dict input) or `param_0`, `param_1`, ...
            (array input).
        aggregation    : callable or None, optional, default: None
            Controls both the posterior layout and the target summary. If
            None: one panel per (dataset, parameter) pair. If a callable
            (e.g. np.mean, np.median): posterior samples (and `targets`,
            if given) are pooled/aggregated across datasets into one
            panel per parameter.
        mixture_names  : dict or None, optional, default: None
            Mapping from parameter name to a list of component names.
            Defaults to `self.simulator.prior._mixture_names()` when not
            supplied.
        **kwargs
            Forwarded to `plot_time_invariant_posterior` (e.g.
            `num_cols`, `color`, `title_fontsize`, `label_fontsize`,
            `tick_fontsize`, `figsize`).

        Returns
        -------
        fig : plt.Figure - the figure instance for optional saving
        """
        if variable_keys is None:
            variable_keys = self.simulator.hyper_keys + self.simulator.shared_keys
        if mixture_names is None:
            mixture_names = self.simulator.prior._mixture_names()

        return plot_time_invariant_posterior(
            estimates=estimates,
            targets=targets,
            variable_keys=variable_keys,
            variable_names=variable_names,
            aggregation=aggregation,
            mixture_names=mixture_names,
            **kwargs,
        )
