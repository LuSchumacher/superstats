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
    """Lightweight amortized Bayesian inference workflow wrapper."""


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
        if self.checkpoint_filepath is None:
            return
        path = os.path.join(self.checkpoint_filepath, "history.pkl")
        if not os.path.exists(path):
            return
        with open(path, "rb") as f:
            self.workflow.history = pickle.load(f)


    def _save_history(self, new_history: keras.callbacks.History) -> None:
        if self.checkpoint_filepath is None:
            return
        existing = self.workflow.history
        if existing is not None:
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
        return self.workflow.history

    @property
    def approximator(self):
        return self.workflow.approximator


    def sample(
        self,
        data: np.ndarray,
        num_samples: int = 1000,
        inference_batch_size: int = 5,
        **kwargs,
    ) -> dict[str, np.ndarray]:
        """
        Run inference on observed data.

        Parameters
        ----------
        data : np.ndarray, shape (num_datasets, num_steps, data_dims)
        num_samples : int
            Number of posterior samples per dataset.
        inference_batch_size : int
            Datasets per GPU batch to avoid OOM.

        Returns
        -------
        dict of {param_name: np.ndarray}
        """

        samples = self.approximator.sample(
            conditions={"data": data},
            num_samples=num_samples,
            batch_size=inference_batch_size,
            **kwargs,
        )
        return samples

    def plot_history(self, history):
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
        """
        Simulate, run inference, and plot time-varying parameter recovery.

        Parameters
        ----------
        num_steps : int
        num_sims : int
        num_samples : int
        inference_batch_size : int
        bootstrap_calibration : bool
        n_bootstrap : int
        param_names : list of str, optional
        **plot_kwargs
            Passed to plot_time_varying_validation.
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
        """
        Plot time-invariant parameter recovery and calibration.

        Parameters
        ----------
        true_params : dict
            {param_name: np.ndarray of shape (num_sims, dim)}
        samples : dict
            {param_name: np.ndarray of shape (num_sims, num_samples, steps, dim)}
        param_names : list of str, optional
        num_out : int, optional
            Number of aggregated samples per simulation. Defaults to num_samples.
        rng : optional
            Seed or numpy Generator for reproducibility.
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
        return plot_time_invariant_posterior(
            samples=samples,
            hyper_keys=self.simulator.hyper_keys,
            shared_keys=self.simulator.shared_keys,
            mixture_names=self.simulator.prior._mixture_names(),
            **kwargs,
        )

    def plot_joint_posterior(
        self,
        samples: dict,
        **kwargs,
    ):
        return plot_joint_posterior(
            samples=samples,
            local_keys=self.simulator.local_keys,
            hyper_keys=self.simulator.hyper_keys,
            shared_keys=self.simulator.shared_keys,
            mixture_names=self.simulator.prior._mixture_names(),
            **kwargs,
        )