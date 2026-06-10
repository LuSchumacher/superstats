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

PALETTE = ["#C1440E", "#E8871A", "#D4A843", "#7B3F00"]

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
            self.local_keys = self.simulator.local_keys
            self.hyper_keys = self.simulator.hyper_keys
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

        self.workflow = bf.BasicWorkflow(
            simulator=self.simulator,
            adapter=self.adapter,
            summary_network=self.summary_net,
            inference_network=self.inference_net,
            standardize="all",
            checkpoint_filepath=self.checkpoint_filepath
        )
    
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
            tile_to_steps=True
        )
        test_data = self.simulator.sample(
            batch_size=test_data_size,
            steps=steps,
            tile_to_steps=True
        )
        self.history = self.workflow.fit_offline(
            data=train_data,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=test_data
        )
        if self.checkpoint_filepath is not None:
            self.approximator = keras.saving.load_model(f"{self.checkpoint_filepath}/model.keras")
        
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
            self.approximator = keras.saving.load_model(f"{self.checkpoint_filepath}/model.keras")

        return self.history
    
    def plot_history(self):
        if self.history is None:
            raise ValueError("No training history found. Please run fit_offline or fit_online first.")
        return bf.diagnostics.plots.loss(self.history, train_color="#822621")
    
    def validate_time_varying(
        self,
        steps: int = 100,
        num_sims: int = 200,
        num_samples: int = 100,
        bootstrap_calibration: bool = False,
        n_bootstrap: int = 1000,
        param_names: list | None = None,
        **plot_kwargs,
    ):
        """
        Simulate, fit, and plot time-varying parameter recovery diagnostics.

        Parameters
        ----------
        steps : int
            Number of time steps per simulation.
        num_sims : int
            Number of datasets to simulate.
        num_samples : int
            Number of posterior samples per dataset.
        bootstrap_calibration : bool
            Whether to bootstrap calibration CI.
        n_bootstrap : int
            Number of bootstrap samples.
        param_names : list of str, optional
            Display names for local parameters.
        **plot_kwargs
            Passed to plot_time_varying_validation.
        """
        # -- simulate --
        validation_data = self.simulator.sample(
            batch_size=num_sims,
            steps=steps,
            tile_to_steps=True,
        )

        # -- posterior samples --
        samples = self.approximator.sample(
            conditions={"data": validation_data["data"]},
            num_samples=num_samples,
        )

        # -- extract local params --
        local_keys = self.simulator.local_keys

        # true: (num_sims, steps, num_params)
        true = np.stack(
            [validation_data[k][..., 0] for k in local_keys],
            axis=-1,
        )

        # estimated: (num_sims, num_samples, steps, num_params)
        estimated = np.concatenate(
            [samples[k] for k in local_keys],
            axis=-1,
        )

        # transpose to (num_sims, steps, num_samples, num_params)
        estimated = estimated.transpose(0, 2, 1, 3)

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
        param_names: list | None = None,
        recovery_kwargs: dict | None = None,
        calibration_kwargs: dict | None = None,
    ):
        """
        Simulate and plot time-invariant parameter recovery and calibration.

        Parameters
        ----------
        steps : int
            Number of time steps per simulation.
        num_sims : int
            Number of datasets to simulate.
        num_samples : int
            Number of posterior samples per dataset.
        param_names : list of str, optional
            Display names for parameters. Auto-expanded for mixture weights.
        recovery_kwargs : dict, optional
            Passed to bf.diagnostics.plots.recovery.
        calibration_kwargs : dict, optional
            Passed to bf.diagnostics.plots.calibration_ecdf.

        Returns
        -------
        fig_recovery : plt.Figure
        fig_calibration : plt.Figure
        """
        recovery_kwargs    = recovery_kwargs    or {}
        calibration_kwargs = calibration_kwargs or {}

        # -- simulate --
        validation_data = self.simulator.sample(
            batch_size=num_sims,
            steps=steps,
            tile_to_steps=True,
        )

        # -- posterior samples --
        samples = self.approximator.sample(
            conditions={"data": validation_data["data"]},
            num_samples=num_samples,
        )

        # -- extract shared + hyper keys --
        keys = self.simulator.hyper_keys + self.simulator.shared_keys

        if not keys:
            raise ValueError("No time-invariant parameters found.")

        # -- build targets and estimates with expanded mixture weights --
        target_list   = []
        estimate_list = []
        expanded_names = []

        for k in keys:
            t_arr = validation_data[k][:, 0, :]  # (num_sims, dim)
            e_arr = samples[k][:, :, 0, :]       # (num_sims, num_samples, dim)

            dim = t_arr.shape[-1]
            if dim > 1:
                # mixture weights — expand into one column per component
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

        targets   = np.concatenate(target_list,   axis=-1)  # (num_sims, num_params)
        estimates = np.concatenate(estimate_list, axis=-1)  # (num_sims, num_samples, num_params)

        if param_names is None:
            param_names = expanded_names

        fig_recovery = bf.diagnostics.plots.recovery(
            estimates=estimates,
            targets=targets,
            variable_names=param_names,
            color="#822621",
            **recovery_kwargs,
        )

        fig_calibration = bf.diagnostics.plots.calibration_ecdf(
            estimates=estimates,
            targets=targets,
            variable_names=param_names,
            rank_ecdf_color="#822621",
            **calibration_kwargs,
        )

        return fig_recovery, fig_calibration