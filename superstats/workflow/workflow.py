from typing import Literal

import bayesflow as bf
import numpy as np
import keras
import functools
import os
import pickle
import warnings

from bayesflow.adapters import Adapter
from bayesflow.networks import SummaryNetwork, InferenceNetwork

from superstats.simulation import GenerativeModel
from superstats.networks import RecurrentNet
from superstats.defaults.network_defaults import (
    DEFAULT_SUMMARY_NETWORK,
    DEFAULT_INFERENCE_NETWORK,
)
from superstats.diagnostics.plots.time_varying_validation import (
    plot_time_varying_validation,
)
from superstats.diagnostics.plots.posterior_samples import (
    plot_time_varying_posterior,
    plot_time_invariant_posterior,
    # plot_joint_posterior,
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
        **kwargs
    ):
        self.simulator = simulator

        if summary_network == "recurrent":
            self.summary_network = RecurrentNet(**DEFAULT_SUMMARY_NETWORK)
        else:
            self.summary_network = summary_network

        if inference_network == "consistency":
            self.inference_network = bf.networks.StableConsistencyModel(
                **DEFAULT_INFERENCE_NETWORK
            )
        else:
            self.inference_network = inference_network

        if adapter is not None:
            self.adapter = adapter
        else:
            self.local_keys  = self.simulator.local_keys
            self.hyper_keys  = self.simulator.hyper_keys
            self.shared_keys = self.simulator.shared_keys
            self.adapter = (
                bf.Adapter()
                    .convert_dtype("float64", "float32")
                    .concatenate(
                        self.local_keys + self.hyper_keys + self.shared_keys,
                        into="inference_variables"
                    )
                    .rename("data", "summary_variables")
            )

        self.checkpoint_filepath = checkpoint_filepath

        if restore_approximator and self.checkpoint_filepath is not None and os.path.isdir(self.checkpoint_filepath):
            restore = True
        elif restore_approximator:
            warnings.warn(
                f"restore_approximator=True but no model found at '{self.checkpoint_filepath}'. "
                "Starting with no trained model.",
                stacklevel=2
            )
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
            **kwargs
        )


        if restore_history and self.checkpoint_filepath is not None and os.path.isdir(self.checkpoint_filepath):
            self._load_history()
        elif restore_history:
            warnings.warn(
                f"restore_history=True but no history found at '{self.checkpoint_filepath}'. "
                "Starting with no history.",
                stacklevel=2
            )


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
        self,
        data,
        validation_data,
        epochs: int = 100,
        batch_size: int = 32,
        save_history: bool = True,
        **kwargs
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
            data=data,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            **kwargs
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
        **kwargs
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
        self.simulator.sample = functools.partial(
            original_sample, num_steps=num_steps, tile_to_steps=True
        )
        try:
            history = self.workflow.fit_online(
                epochs=epochs,
                num_batches_per_epoch=num_batches_per_epoch,
                batch_size=batch_size,
                **kwargs
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
        data: np.ndarray,
        num_samples: int = 1000,
        inference_batch_size: int = 5,
        **kwargs,
    ) -> dict[str, np.ndarray]:
        """Run inference on observed data.

        Parameters
        ----------
        data                 : np.ndarray of shape (num_datasets, num_steps, data_dims)
            Observed data to condition on.
        num_samples          : int, optional, default: 1000
            Number of posterior samples per dataset.
        inference_batch_size : int, optional, default: 5
            Datasets per GPU batch, to avoid out-of-memory errors.
        **kwargs
            Forwarded to `self.approximator.sample`.

        Returns
        -------
        samples : dict of {param_name: np.ndarray} - posterior samples
            per parameter
        """
        samples = self.approximator.sample(
            conditions={"data": data},
            num_samples=num_samples,
            batch_size=inference_batch_size,
            **kwargs,
        )
        return samples

    def resimulate_posterior(
        self,
        posterior_samples: dict[str, np.ndarray],
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
                "Posterior sample arrays must have at least 3 dimensions: "
                "(batch_size, num_samples, num_steps, ...)."
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
                    f"Posterior parameter '{name}' has batch size {arr.shape[0]} "
                    f"but expected {batch_size}."
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
                raise ValueError(
                    f"Cannot reshape posterior parameter '{name}' with shape {arr.shape}."
                )

        for name, value in fixed_params.items():
            expanded_params[name] = np.broadcast_to(
                np.asarray(value),
                (batch_size * num_sims,)
            )

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

    def validate_time_varying(
        self,
        true_params: dict,
        samples: dict,
        bootstrap_calibration: bool = False,
        num_bootstrap: int = 1000,
        param_names: list | None = None,
        **plot_kwargs,
    ):
        """Simulate, run inference, and plot time-varying parameter recovery.

        Parameters
        ----------
        true_params           : dict
            Ground-truth local parameter trajectories, keyed by
            parameter name; each value has shape
            (batch_size, num_steps, 1).
        samples               : dict
            Posterior samples for the same parameters, keyed by name;
            each value has shape
            (batch_size, num_post_samples, num_steps, 1).
        bootstrap_calibration : bool, optional, default: False
            If True, show a bootstrap uncertainty band for the
            calibration error.
        num_bootstrap         : int, optional, default: 1000
            Number of bootstrap resamples. Only used when
            `bootstrap_calibration=True`.
        param_names           : list of str or None, optional, default: None
            Column labels. Defaults to `self.simulator.local_keys` when
            not supplied.
        **plot_kwargs
            Forwarded to `plot_time_varying_validation`.

        Returns
        -------
        fig : plt.Figure - the figure instance for optional saving
        """
        local_keys = self.simulator.local_keys

        true = np.stack(
            [true_params[k][..., 0] for k in local_keys],
            axis=-1,
        )
        estimated = np.concatenate(
            [samples[k] for k in local_keys],
            axis=-1,
        )
        estimated = estimated.transpose(0, 2, 1, 3)

        if param_names is None:
            param_names = local_keys

        return plot_time_varying_validation(
            true=true,
            estimated=estimated,
            param_names=param_names,
            bootstrap_calibration=bootstrap_calibration,
            n_bootstrap=num_bootstrap,
            **plot_kwargs,
        )

    def validate_time_invariant(
        self,
        true_params,
        samples,
        param_names: list | None = None,
        num_out: int | None = None,
        rng=None,
        title_fontsize: int = 22,
        label_fontsize: int = 18,
        metric_fontsize: int = 18,
        tick_fontsize: int = 16,
        color: str = "#822621"
    ):
        """Plot time-invariant parameter recovery and calibration.

        Parameters
        ----------
        true_params     : dict
            Mapping from parameter name to an np.ndarray of shape
            (num_sims, dim).
        samples         : dict
            Mapping from parameter name to an np.ndarray of shape
            (num_sims, num_samples, steps, dim).
        param_names     : list of str or None, optional, default: None
            Names for each (expanded, per-component) column. Defaults
            to the hyper/shared keys, expanded per mixture component
            where applicable.
        num_out         : int or None, optional, default: None
            Number of aggregated samples per simulation. Defaults to
            `num_samples` (no subsampling) when not supplied.
        rng             : int or np.random.Generator or None, optional, default: None
            Seed or generator for reproducible pooling.
        title_fontsize  : int, optional, default: 22
            The font size of the panel titles.
        label_fontsize  : int, optional, default: 18
            The font size of the axis label texts.
        metric_fontsize : int, optional, default: 18
            The font size of the displayed recovery/calibration metric
            text.
        tick_fontsize   : int, optional, default: 16
            The font size of the axis tick labels.
        color           : str, optional, default: "#822621"
            Base plotting color for both figures.

        Returns
        -------
        figs : tuple - `(fig_recovery, fig_calibration)`, the recovery
            and calibration diagnostic figures

        Raises
        ------
        ValueError
            If no time-invariant parameters (`hyper_keys` +
            `shared_keys`) are found on `self.simulator`.
        """
        rng = np.random.default_rng(rng)

        keys = self.simulator.hyper_keys + self.simulator.shared_keys
        if not keys:
            raise ValueError("No time-invariant parameters found.")

        target_list    = []
        estimate_list  = []
        expanded_names = []

        for k in keys:
            t_arr = true_params[k]
            e_arr = samples[k]
            B, S, T, dim = e_arr.shape

            n_out  = num_out if num_out is not None else S
            pooled = e_arr.reshape(B, S * T, dim)
            idx    = rng.integers(0, S * T, size=(B, n_out))
            e_agg  = pooled[np.arange(B)[:, None], idx, :]

            if dim > 1:
                param_key   = k.split("_mixture_weights")[0]
                mixture_obj = self.simulator.prior.params.get(param_key)
                if hasattr(mixture_obj, "names") and len(mixture_obj.names) == dim:
                    comp_names = [f"{k}_{n}" for n in mixture_obj.names]
                else:
                    comp_names = [f"{k}_{i}" for i in range(dim)]

                for i, name in enumerate(comp_names):
                    target_list.append(t_arr[:, i:i+1])
                    estimate_list.append(e_agg[:, :, i:i+1])
                    expanded_names.append(name)
            else:
                target_list.append(t_arr)
                estimate_list.append(e_agg)
                expanded_names.append(k)

        targets   = np.concatenate(target_list,   axis=-1)
        estimates = np.concatenate(estimate_list, axis=-1)

        if param_names is None:
            param_names = expanded_names

        fig_recovery = bf.diagnostics.plots.recovery(
            estimates=estimates,
            targets=targets,
            variable_names=param_names,
            label_fontsize=label_fontsize,
            title_fontsize=title_fontsize,
            metric_fontsize=metric_fontsize,
            tick_fontsize=tick_fontsize,
            color=color,
        )

        fig_calibration = bf.diagnostics.plots.calibration_ecdf(
            estimates=estimates,
            targets=targets,
            variable_names=param_names,
            label_fontsize=label_fontsize,
            title_fontsize=title_fontsize,
            metric_fontsize=metric_fontsize,
            tick_fontsize=tick_fontsize,
            rank_ecdf_color=color,
        )

        return fig_recovery, fig_calibration


    def plot_time_varying_posterior(
        self,
        samples: dict,
        **kwargs,
    ):
        """Plot time-varying posterior diagnostics.

        Parameters
        ----------
        samples : dict
            Posterior sample dictionary.
        **kwargs
            Forwarded to `plot_time_varying_posterior`.

        Returns
        -------
        fig : plt.Figure - the figure instance for optional saving
        """
        return plot_time_varying_posterior(
            samples=samples,
            local_keys=self.simulator.local_keys,
            **kwargs,
        )

    def plot_time_invariant_posterior(
        self,
        samples: dict,
        **kwargs,
    ):
        """Plot time-invariant posterior diagnostics.

        Parameters
        ----------
        samples : dict
            Posterior sample dictionary.
        **kwargs
            Forwarded to `plot_time_invariant_posterior`.

        Returns
        -------
        fig : plt.Figure - the figure instance for optional saving
        """
        return plot_time_invariant_posterior(
            samples=samples,
            hyper_keys=self.simulator.hyper_keys,
            shared_keys=self.simulator.shared_keys,
            mixture_names=self.simulator.prior._mixture_names(),
            **kwargs,
        )

    # def plot_joint_posterior(
    #     self,
    #     samples: dict,
    #     **kwargs,
    # ):
    #     return plot_joint_posterior(
    #         samples=samples,
    #         local_keys=self.simulator.local_keys,
    #         hyper_keys=self.simulator.hyper_keys,
    #         shared_keys=self.simulator.shared_keys,
    #         mixture_names=self.simulator.prior._mixture_names(),
    #         **kwargs,
    #     )