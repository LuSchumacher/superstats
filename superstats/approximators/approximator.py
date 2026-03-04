from collections.abc import Mapping, Sequence
from typing import Literal, Tuple

import numpy as np

import bayesflow as bf
import keras

from bayesflow.adapters import Adapter
from bayesflow.networks import InferenceNetwork, SummaryNetwork
from bayesflow.types import Tensor
from bayesflow.utils.serialization import serialize, serializable

from .approximator import Approximator, ContinuousApproximator



@serializable("bayesflow.approximators")
class SuperstatsApproximator(Approximator):
    """
    TODO

    Parameters
    ----------
    adapter : bayesflow.adapters.Adapter
        Adapter for data processing. You can use :py:meth:`build_adapter`
        to create it.
    inference_network : InferenceNetwork
        The inference network used for posterior or likelihood approximation.
    summary_network : SummaryNetwork, optional
        The summary network used for data summarization (default is None).
    standardize : str | Sequence[str] | None
        The variables to standardize before passing to the networks. Can be either
        "all" or any subset of ["inference_variables", "summary_variables", "inference_conditions"].
        (default is "inference_variables").
    **kwargs : dict, optional
        Additional arguments passed to the :py:class:`bayesflow.approximators.Approximator` class.
    """

    def __init__(
        self,
        *,
        adapter: bf.Adapter,
        local_inference_network: InferenceNetwork,
        global_inference_network: InferenceNetwork,
        local_summary_network: SummaryNetwork = None,
        global_summary_network: SummaryNetwork = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.adapter = adapter
        
        self.local_approximator = ContinuousApproximator(
            inference_network=local_inference_network,
            summary_network=local_summary_network,
            adapter=None
        )

        self.shared_global_approximator = ContinuousApproximator(
            inference_network=global_inference_network,
            summary_network=global_summary_network,
            global_adapter=None
        )

        self.has_distribution = True

    def compute_metrics(
        self,
        inference_variables: Tensor,
        inference_conditions: Tensor = None,
        summary_variables: Tensor = None,
        sample_weight: Tensor = None,
        stage: str = "training",
    ) -> dict[str, Tensor]:
        """
        Computes loss and tracks metrics for the inference and summary networks.

        This method orchestrates the end-to-end computation of metrics and loss for a model
        with both inference and optional summary network. It handles standardization of input
        variables, combines summary outputs with inference conditions when necessary, and
        aggregates loss and all tracked metrics into a unified dictionary. The returned dictionary
        includes both the total loss and all individual metrics, with keys indicating their source.

        Parameters
        ----------
        inference_variables : Tensor
            Input tensor(s) for the inference network. These are typically latent variables to be modeled.
        inference_conditions : Tensor, optional
            Conditioning variables for the inference network (default is None).
            May be combined with outputs from the summary network if present.
        summary_variables : Tensor, optional
            Input tensor(s) for the summary network (default is None). Required if
            a summary network is present.
        sample_weight : Tensor, optional
            Weighting tensor for metric computation (default is None).
        summary_attention_mask : Tensor, optional
            Attention mask forwarded to the summary network (default is None).
        summary_mask : Tensor, optional
            Padding / key mask forwarded to the summary network (default is None).
        inference_attention_mask : Tensor, optional
            Attention mask forwarded to the inference network (default is None).
        inference_mask : Tensor, optional
            Padding / key mask forwarded to the inference network (default is None).
        stage : str, optional
            Current training stage (e.g., "training", "validation", "inference"). Controls
            the behavior of standardization and some metric computations (default is "training").

        Returns
        -------
        metrics : dict[str, Tensor]
            Dictionary containing the total loss under the key "loss", as well as all tracked
            metrics for the inference and summary networks. Each metric key is prefixed with
            "inference_" or "summary_" to indicate its source.
        """

        local_metrics = self.local_approximator.compute_metrics(
            inference_variables=inference_variables,
            inference_conditions=inference_conditions,
            summary_variables=summary_variables,
            sample_weight=sample_weight,
            stage=stage,
        )

        global_metrics = self.shared_global_approximator.compute_metrics(
            inference_variables=inference_variables,
            inference_conditions=inference_conditions,
            summary_variables=summary_variables,
            sample_weight=sample_weight,
            stage=stage,
        )

        metrics = {}

        metrics = {f"local/{metric_key}": value for metric_key, value in local_metrics.items()}
        metrics.update({f"global/{metric_key}": value for metric_key, value in global_metrics.items()})

        losses = [v for k, v in metrics.items() if "loss" in k]
        metrics["loss"] = keras.ops.sum(losses)

        return metrics

    def fit(self, *args, **kwargs):
        """
        Trains the approximator on the provided dataset or on-demand data generated from the given simulator.
        If `dataset` is not provided, a dataset is built from the `simulator`.
        If the model has not been built, it will be built using a batch from the dataset.

        Parameters
        ----------
        dataset : keras.utils.PyDataset, optional
            A dataset containing simulations for training. If provided, `simulator` must be None.
        simulator : Simulator, optional
            A simulator used to generate a dataset. If provided, `dataset` must be None.
        **kwargs
            Additional keyword arguments passed to `keras.Model.fit()`, as described in:

        https://github.com/keras-team/keras/blob/v3.13.2/keras/src/backend/tensorflow/trainer.py#L314

        Returns
        -------
        keras.callbacks.History
            A history object containing the training loss and metrics values.

        Raises
        ------
        ValueError
            If both `dataset` and `simulator` are provided or neither is provided.
        """
        return super().fit(*args, **kwargs, adapter=self.adapter)

    def get_config(self):
        return super().get_config()

    def sample(
        self,
        *,
        num_samples: int,
        conditions: Mapping[str, np.ndarray],
        batch_size: int | None = None,
        sample_shape: Literal["infer"] | Tuple[int] | int = "infer",
        return_summaries: bool = False,
        **kwargs,
    ) -> dict[str, np.ndarray]:
        """
        Generates samples from the approximator given input conditions. The `conditions` dictionary is preprocessed
        using the `adapter`. Samples are converted to NumPy arrays after inference.

        Parameters
        ----------
        num_samples : int
            Number of samples to generate.
        conditions : dict[str, np.ndarray]
            Dictionary of conditioning variables as NumPy arrays.
        split : bool, default=False
            Whether to split the output arrays along the last axis and return one sample array per target variable.
        batch_size : int or None, optional
            If provided, the conditions are split into batches of size `batch_size`, for which samples are generated
            sequentially. Can help with memory management for large sample sizes.
        sample_shape : str or tuple of int, optional
            Trailing structural dimensions of each generated sample, excluding the batch and target (intrinsic)
            dimension. For example, use `(time,)` for time series or `(height, width)` for images.

            If set to `"infer"` (default), the structural dimensions are inferred from the `inference_conditions`.
            In that case, all non-vector dimensions except the last (channel) dimension are treated as structural
            dimensions. For example, if the final `inference_conditions` have shape `(batch_size, time, channels)`,
            then `sample_shape` is inferred as `(time,)`, and the generated samples will have shape
            `(num_conditions, num_samples, time, target_dim)`.
        return_summaries: bool, optional
            If set to True and a summary network is present, will return the learned summary statistics for
            the provided conditions.
        **kwargs : dict
            Additional keyword arguments for the sampling process.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary containing generated samples with the same keys as `conditions`.
        """

        conditions = conditions.copy()
    
        global_samples = self.shared_global_approximator.sample(
            num_samples=num_samples,
            conditions=conditions,
            batch_size=batch_size,
            sample_shape=sample_shape,
            return_summaries=return_summaries,
            **kwargs,
        )

        conditions |= global_samples

        # Fix shapes: TODO

        local_samples = self.local_approximator.sample(
            num_samples=num_samples,
            conditions=conditions,
            batch_size=batch_size,
            sample_shape=sample_shape,
            return_summaries=return_summaries,
            **kwargs,
        )

        return local_samples | global_samples

    def log_prob(self, data: Mapping[str, np.ndarray], **kwargs) -> np.ndarray:
        """
        Computes the log-probability of given data under the model. The `data` dictionary is preprocessed using the
        `adapter`. Log-probabilities are returned as NumPy arrays.

        Parameters
        ----------
        data : Mapping[str, np.ndarray]
            Dictionary of observed data as NumPy arrays.
        **kwargs : dict
            Additional keyword arguments for the adapter and log-probability computation.

        Returns
        -------
        np.ndarray
            Log-probabilities of the distribution `p(inference_variables | inference_conditions, h(summary_conditions))`
        """

        log_prob_global = self.shared_global_approximator.log_prob(data=data, **kwargs)
        log_probs_local = self.local_approximator.log_prob(data=data, **kwargs)

        return log_prob_global + log_probs_local