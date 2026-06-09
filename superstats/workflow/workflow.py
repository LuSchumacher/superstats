from __future__ import annotations
from typing import Optional

import bayesflow as bf
import keras
import functools

from superstats.defaults.network_defaults import (
    DEFAULT_SUMMARY_NETWORK,
    DEFAULT_INFERENCE_NETWORK,
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
        return bf.diagnostics.plots.loss(self.history)