from typing import Optional

import bayesflow as bf
import numpy as np
import keras
import functools

from superstats.defaults.network_defaults import (
    DEFAULT_SUMMARY_NETWORK,
    DEFAULT_INFERENCE_NETWORK,
)

from superstats.diagnostics.plots.time_varying_validation import (
    plot_time_varying_validation as _plot_time_varying_validation,
)

from superstats.diagnostics.plots.posterior_samples import (
    plot_time_varying_posterior  as _plot_time_varying_posterior,
    plot_time_invariant_posterior as _plot_time_invariant_posterior,
    plot_joint_posterior          as _plot_joint_posterior,
)


class Workflow:
    """Lightweight amortized Bayesian inference workflow wrapper."""

    def __init__(
        self,
        simulator: Optional[object] = None,
        adapter: Optional[object] = None,
        summary_net: Optional[object] = None,
        inference_net: Optional[object] = None,
        checkpoint_filepath: Optional[str] = None,
    ):
        self.simulator = simulator
        self.summary_net = summary_net if summary_net is not None else DEFAULT_SUMMARY_NETWORK
        self.inference_net = inference_net if inference_net is not None else DEFAULT_INFERENCE_NETWORK

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
        self.history             = None
        self.approximator        = None

        self.workflow = bf.BasicWorkflow(
            simulator=self.simulator,
            adapter=self.adapter,
            summary_network=self.summary_net,
            inference_network=self.inference_net,
            standardize="all",
            checkpoint_filepath=self.checkpoint_filepath
        )

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _sample_posterior(
        self,
        data: np.ndarray,
        num_samples: int,
        inference_batch_size: int,
    ) -> dict:
        """
        Run posterior inference in batches to avoid OOM.

        Parameters
        ----------
        data : np.ndarray, shape (num_datasets, num_steps, data_dims)
        num_samples : int
        inference_batch_size : int

        Returns
        -------
        dict of {param_name: np.ndarray}
        """
        num_datasets = data.shape[0]
        all_samples  = []

        for start in range(0, num_datasets, inference_batch_size):
            end   = min(start + inference_batch_size, num_datasets)
            batch = data[start:end]
            batch_samples = self.approximator.sample(
                conditions={"data": batch},
                num_samples=num_samples,
            )
            all_samples.append(batch_samples)

        return {
            k: np.concatenate([s[k] for s in all_samples], axis=0)
            for k in all_samples[0].keys()
        }

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------

    def fit_offline(
        self,
        steps: int,
        train_data_size: int = 20000,
        test_data_size: int = 250,
        epochs: int = 100,
        batch_size: int = 32,
    ):
        train_data = self.simulator.sample(
            batch_size=train_data_size,
            steps=steps,
            tile_to_steps=True,
        )
        test_data = self.simulator.sample(
            batch_size=test_data_size,
            steps=steps,
            tile_to_steps=True,
        )
        self.history = self.workflow.fit_offline(
            data=train_data,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=test_data,
        )
        if self.checkpoint_filepath is not None:
            self.approximator = keras.saving.load_model(
                f"{self.checkpoint_filepath}/model.keras"
            )

        return self.history

    def fit_online(
        self,
        steps: int,
        epochs: int = 100,
        num_batches_per_epoch: int = 100,
        batch_size: int = 32,
    ):
        original_sample = self.simulator.sample
        self.simulator.sample = functools.partial(
            original_sample, steps=steps, tile_to_steps=True
        )
        try:
            self.history = self.workflow.fit_online(
                epochs=epochs,
                num_batches_per_epoch=num_batches_per_epoch,
                batch_size=batch_size,
            )
        finally:
            self.simulator.sample = original_sample

        if self.checkpoint_filepath is not None:
            self.approximator = keras.saving.load_model(
                f"{self.checkpoint_filepath}/model.keras"
            )

        return self.history

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------

    def fit_data(
        self,
        data: np.ndarray,
        num_samples: int = 1000,
        inference_batch_size: int = 10,
    ) -> dict:
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
        return self._sample_posterior(data, num_samples, inference_batch_size)

    # ------------------------------------------------------------------
    # diagnostics
    # ------------------------------------------------------------------

    def plot_history(self):
        if self.history is None:
            raise ValueError("No training history. Run fit_offline or fit_online first.")
        return bf.diagnostics.plots.loss(self.history, train_color="#822621")

    def validate_time_varying(
        self,
        steps: int = 100,
        num_sims: int = 200,
        num_samples: int = 100,
        inference_batch_size: int = 10,
        bootstrap_calibration: bool = False,
        n_bootstrap: int = 1000,
        param_names: list | None = None,
        **plot_kwargs,
    ):
        """
        Simulate, run inference, and plot time-varying parameter recovery.

        Parameters
        ----------
        steps : int
        num_sims : int
        num_samples : int
        inference_batch_size : int
        bootstrap_calibration : bool
        n_bootstrap : int
        param_names : list of str, optional
        **plot_kwargs
            Passed to plot_time_varying_validation.
        """
        validation_data = self.simulator.sample(
            batch_size=num_sims,
            steps=steps,
            tile_to_steps=True,
        )

        samples = self._sample_posterior(
            validation_data["data"], num_samples, inference_batch_size
        )

        local_keys = self.simulator.local_keys

        true = np.stack(
            [validation_data[k][..., 0] for k in local_keys],
            axis=-1,
        )  # (num_sims, steps, num_params)

        estimated = np.concatenate(
            [samples[k] for k in local_keys],
            axis=-1,
        )  # (num_sims, num_samples, steps, num_params)

        estimated = estimated.transpose(0, 2, 1, 3)
        # (num_sims, steps, num_samples, num_params)

        if param_names is None:
            param_names = local_keys

        return _plot_time_varying_validation(
            true=true,
            estimated=estimated,
            param_names=param_names,
            bootstrap_calibration=bootstrap_calibration,
            n_bootstrap=n_bootstrap,
            **plot_kwargs,
        )

    def validate_time_invariant(
        self,
        steps: int = 100,
        num_sims: int = 200,
        num_samples: int = 100,
        inference_batch_size: int = 10,
        param_names: list | None = None,
        recovery_kwargs: dict | None = None,
        calibration_kwargs: dict | None = None,
    ):
        """
        Simulate, run inference, and plot time-invariant parameter recovery
        and calibration.

        Parameters
        ----------
        steps : int
        num_sims : int
        num_samples : int
        inference_batch_size : int
        param_names : list of str, optional
        recovery_kwargs : dict, optional
        calibration_kwargs : dict, optional

        Returns
        -------
        fig_recovery : plt.Figure
        fig_calibration : plt.Figure
        """
        recovery_kwargs = {
            "label_fontsize":  14,
            "title_fontsize":  16,
            "metric_fontsize": 14,
            "tick_fontsize":   12,
            "color":           "#822621",
        } | (recovery_kwargs or {})

        calibration_kwargs = {
            "label_fontsize":  14,
            "title_fontsize":  16,
            "legend_fontsize": 12,
            "tick_fontsize":   12,
            "rank_ecdf_color": "#822621",
        } | (calibration_kwargs or {})

        validation_data = self.simulator.sample(
            batch_size=num_sims,
            steps=steps,
            tile_to_steps=True,
        )

        samples = self._sample_posterior(
            validation_data["data"], num_samples, inference_batch_size
        )

        keys = self.simulator.hyper_keys + self.simulator.shared_keys

        if not keys:
            raise ValueError("No time-invariant parameters found.")

        target_list    = []
        estimate_list  = []
        expanded_names = []

        for k in keys:
            t_arr = validation_data[k][:, 0, :]  # (num_sims, dim)
            e_arr = samples[k][:, :, 0, :]       # (num_sims, num_samples, dim)
            dim   = t_arr.shape[-1]

            if dim > 1:
                param_key   = k.split("_mixture_weights")[0]
                mixture_obj = self.simulator.prior.params.get(param_key)
                if hasattr(mixture_obj, "names") and len(mixture_obj.names) == dim:
                    comp_names = [f"{k}_{n}" for n in mixture_obj.names]
                else:
                    comp_names = [f"{k}_{i}" for i in range(dim)]

                for i, name in enumerate(comp_names):
                    target_list.append(t_arr[:, i:i+1])
                    estimate_list.append(e_arr[:, :, i:i+1])
                    expanded_names.append(name)
            else:
                target_list.append(t_arr)
                estimate_list.append(e_arr)
                expanded_names.append(k)

        targets   = np.concatenate(target_list,   axis=-1)
        estimates = np.concatenate(estimate_list, axis=-1)

        if param_names is None:
            param_names = expanded_names

        fig_recovery = bf.diagnostics.plots.recovery(
            estimates=estimates,
            targets=targets,
            variable_names=param_names,
            **recovery_kwargs,
        )

        fig_calibration = bf.diagnostics.plots.calibration_ecdf(
            estimates=estimates,
            targets=targets,
            variable_names=param_names,
            **calibration_kwargs,
        )

        return fig_recovery, fig_calibration

    # ------------------------------------------------------------------
    # posterior plots
    # ------------------------------------------------------------------

    def plot_time_varying_posterior(
        self,
        samples: dict,
        **kwargs,
    ):
        return _plot_time_varying_posterior(
            samples=samples,
            local_keys=self.simulator.local_keys,
            **kwargs,
        )

    def plot_time_invariant_posterior(
        self,
        samples: dict,
        **kwargs,
    ):
        return _plot_time_invariant_posterior(
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
        return _plot_joint_posterior(
            samples=samples,
            local_keys=self.simulator.local_keys,
            hyper_keys=self.simulator.hyper_keys,
            shared_keys=self.simulator.shared_keys,
            mixture_names=self.simulator.prior._mixture_names(),
            **kwargs,
        )