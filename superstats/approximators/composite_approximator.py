"""Shared composition of a global posterior and a conditional sequence posterior."""

from collections.abc import Mapping, Sequence

import keras
import numpy as np

from bayesflow.adapters import Adapter
from bayesflow.approximators import Approximator, AutoregressiveApproximator, ContinuousApproximator
from bayesflow.networks import RecurrentNetwork
from bayesflow.networks.decoders import RecurrentDecoder, TransformerDecoder
from bayesflow.networks.helpers import Standardization
from bayesflow.utils import filter_kwargs, split_arrays
from bayesflow.utils.keras_utils import resolve_seed
from bayesflow.utils.serialization import serializable, serialize

from superstats.defaults import (
    DEFAULT_AUTOREGRESSIVE_DECODER_NETWORK,
    DEFAULT_FILTERING_DECODER_NETWORK,
)


@serializable("superstats.approximators")
class CompositeApproximator(Approximator):
    """Compose invariant and conditional time-varying posterior estimators.

    Parameters
    ----------
    varying_inference_network : str or bayesflow.networks.InferenceNetwork, default: "coupling"
        Conditional density network for the time-varying parameters.
    invariant_inference_network : str or bayesflow.networks.InferenceNetwork, default: "coupling"
        Separate conditional density network for the invariant parameters.
        The two heads must use distinct network instances.
    summary_network : str or keras.layers.Layer, default: "recurrent"
        Observation encoder mapping (B, T, Dx) to (B, T, H), with known
        conditions appended to its inputs when present. The default recurrent
        encoder is bidirectional for smoothing and unidirectional for filtering.
    invariant_pooling : keras.layers.Layer or None, optional, default: None
        Pooling layer mapping encoded observations (B, T, H) to (B, G).
        If None, use a masked mean with log(1 + valid sequence length)
        appended, giving G = H + 1. Requires invariant targets.
    adapter : bayesflow.adapters.Adapter or None, optional, default: None
        Adapter mapping named data to the canonical keys described below.
        If None, use an identity adapter. See `build_adapter` for named inputs.
    mode : {"smoothing", "filtering"}, optional, default: "smoothing"
        Smoothing uses the complete observation sequence for each varying
        factor. Filtering restricts each varying factor to its observation
        prefix. The invariant head always uses the complete sequence.
    standardize : str, sequence of str, or None, optional, default: "all"
        Canonical variables to standardize: "inference_variables",
        "invariant_variables", "summary_variables", and "inference_conditions".
        "all" selects all four; None disables standardization. Each child
        approximator standardizes its own targets.
    decoder_network : keras.layers.Layer or None, optional, default: None
        Decoder used when `joint` is True. Defaults to a TransformerDecoder
        for smoothing or RecurrentDecoder for filtering. A custom filtering
        decoder must be a RecurrentDecoder or declare supports_filtering=True.
        Requires varying targets.
    joint : bool, optional, default: False
        If True, use an autoregressive sequence posterior. Otherwise, use
        conditionally independent per-time posteriors given observations and
        invariant parameters.
    has_varying : bool, optional, default: True
        Enable the time-varying target head.
    has_invariant : bool, optional, default: True
        Enable the invariant target head. At least one head must be enabled.
        Disabled heads require their network argument to retain its default.
    **kwargs
        Additional arguments passed to the BayesFlow Approximator base class.

    Raises
    ------
    TypeError
        If the legacy `inference_network` constructor argument is supplied.
    ValueError
        If the mode, standardization keys, head configuration, network sharing,
        or filtering decoder is invalid.

    Notes
    -----
    Canonical varying targets use "inference_variables" with shape (B, T, Dv),
    invariant targets use "invariant_variables" with shape (B, Di), and
    observations use "summary_variables" with shape (B, T, Dx). B denotes
    datasets, T time steps, and Dv, Di, and Dx the respective feature counts.
    Invariant targets are never tiled over time; they are broadcast only when
    conditioning the varying head.

    Filtering is conditional on the invariant parameters. Since the invariant
    head sees the complete sequence, integrating its draws does not give strict
    online filtering. That requires applying the complete approximator
    separately to each observation prefix.
    """

    def __init__(
        self,
        varying_inference_network="coupling",
        invariant_inference_network="coupling",
        summary_network="recurrent",
        invariant_pooling=None,
        adapter: Adapter | None = None,
        mode: str = "smoothing",
        standardize: str | Sequence[str] | None = "all",
        decoder_network=None,
        joint: bool = False,
        has_varying: bool = True,
        has_invariant: bool = True,
        **kwargs,
    ):
        """Initialize the component networks using the class parameters."""
        if "inference_network" in kwargs:
            raise TypeError("Use varying_inference_network instead of inference_network.")
        super().__init__(**kwargs)

        if mode not in {"smoothing", "filtering"}:
            raise ValueError("mode must be 'smoothing' or 'filtering'.")

        if not has_varying and not has_invariant:
            raise ValueError("At least one target group is required.")
        if not has_varying and not (
            isinstance(varying_inference_network, str) and varying_inference_network == "coupling"
        ):
            raise ValueError("varying_inference_network requires varying targets.")
        if not has_invariant and not (
            isinstance(invariant_inference_network, str) and invariant_inference_network == "coupling"
        ):
            raise ValueError("invariant_inference_network requires invariant targets.")
        if not has_varying and decoder_network is not None:
            raise ValueError("decoder_network requires varying targets.")
        if not has_invariant and invariant_pooling is not None:
            raise ValueError("invariant_pooling requires invariant targets.")
        self.has_varying = has_varying
        self.has_invariant = has_invariant

        from superstats.utils.dispatch import find_embedding_network, find_inference_network

        varying_inference_network = find_inference_network(varying_inference_network) if has_varying else None
        invariant_inference_network = find_inference_network(invariant_inference_network) if has_invariant else None
        if has_varying and has_invariant and varying_inference_network is invariant_inference_network:
            raise ValueError("The two inference heads must be separate network instances.")

        if joint and mode == "filtering" and decoder_network is not None:
            if not isinstance(decoder_network, RecurrentDecoder) and not getattr(
                decoder_network, "supports_filtering", False
            ):
                raise ValueError(
                    "Joint filtering requires RecurrentDecoder or a custom decoder declaring "
                    "supports_filtering=True. TransformerDecoder has unrestricted cross-attention."
                )

        self.mode = mode
        self.joint = joint
        self.adapter = adapter if adapter is not None else Adapter()

        if isinstance(summary_network, str) and summary_network.lower() == "recurrent":
            summary_network = find_embedding_network(summary_network, bidirectional=mode == "smoothing")
        else:
            summary_network = find_embedding_network(summary_network)

        self.summary_network = summary_network
        self.invariant_pooling = invariant_pooling
        self._standardize = standardize
        keys = {"inference_variables", "invariant_variables", "summary_variables", "inference_conditions"}
        selected = (
            keys if standardize == "all" else set([standardize] if isinstance(standardize, str) else standardize or [])
        )
        if selected - keys:
            raise ValueError(f"Unknown standardization variables: {sorted(selected - keys)}.")
        self.standardizer = Standardization(sorted(selected & {"summary_variables", "inference_conditions"}))

        self.invariant_approximator = (
            ContinuousApproximator(
                inference_network=invariant_inference_network,
                standardize="inference_variables" if "invariant_variables" in selected else None,
            )
            if has_invariant
            else None
        )
        sequence_standardize = "inference_variables" if "inference_variables" in selected else None
        if not has_varying:
            self.sequence_approximator = None
        elif joint:
            if decoder_network is None:
                if mode == "filtering":
                    decoder_network = RecurrentDecoder(**DEFAULT_FILTERING_DECODER_NETWORK)
                else:
                    decoder_network = TransformerDecoder(**DEFAULT_AUTOREGRESSIVE_DECODER_NETWORK)
            self.sequence_approximator = AutoregressiveApproximator(
                inference_network=varying_inference_network,
                encoder_network=keras.layers.Identity(),
                decoder_network=decoder_network,
                standardize=sequence_standardize,
            )
        else:
            self.sequence_approximator = ContinuousApproximator(
                inference_network=varying_inference_network, standardize=sequence_standardize
            )
        self.has_distribution = True
        self.seed_generator = keras.random.SeedGenerator()
        self._data_shapes = None

    @property
    def inference_network(self):
        """bayesflow.networks.InferenceNetwork: Varying head, or invariant head if varying targets are disabled."""
        return self.varying_inference_network or self.invariant_inference_network

    @property
    def varying_inference_network(self):
        """bayesflow.networks.InferenceNetwork or None: Density network for enabled varying targets."""
        return self.sequence_approximator.inference_network if self.has_varying else None

    @property
    def invariant_inference_network(self):
        """bayesflow.networks.InferenceNetwork or None: Density network for enabled invariant targets."""
        return self.invariant_approximator.inference_network if self.has_invariant else None

    @classmethod
    def build_adapter(
        cls,
        inference_variables: str | Sequence[str] | None,
        invariant_variables: str | Sequence[str] | None,
        summary_variables: str | Sequence[str],
        inference_conditions: str | Sequence[str] | None = None,
        sample_weight: str | None = None,
        summary_mask: str | None = None,
        summary_attention_mask: str | None = None,
        inference_mask: str | None = None,
        inference_attention_mask: str | None = None,
    ) -> Adapter:
        """Build an adapter for named varying targets, invariant targets, and observations.

        Parameters
        ----------
        inference_variables : str, sequence of str, or None
            Names of varying targets, each shaped (B, T) or (B, T, D).
            Pass None for an absent head and set has_varying=False in the
            constructor.
        invariant_variables : str, sequence of str, or None
            Names of invariant targets, each shaped (B, D), including D=1.
            Pass None for an absent head and set has_invariant=False in the
            constructor. Names must not overlap with varying targets.
        summary_variables : str or sequence of str
            Nonempty observation names, each shaped (B, T) or (B, T, D).
        inference_conditions : str, sequence of str, or None, optional, default: None
            Names of known conditions shaped (B, C) or (B, T, C). Conditions
            are concatenated along their last axis and must have compatible
            leading dimensions.
        sample_weight : str or None, optional, default: None
            Name of dataset weights shaped (B,).
        summary_mask : str or None, optional, default: None
            Name of the observation padding mask shaped (B, T), with True
            indicating valid time steps.
        summary_attention_mask : str or None, optional, default: None
            Name of the attention mask forwarded to the summary network.
        inference_mask : str or None, optional, default: None
            Name of the target padding mask shaped (B, T), with True
            indicating valid time steps for the varying objective.
        inference_attention_mask : str or None, optional, default: None
            Name of the attention mask forwarded to the varying head.

        Returns
        -------
        adapter : bayesflow.adapters.Adapter
            Adapter concatenating each group into its canonical key,
            converting float64 to float32, and preserving only requested keys.
            Inverse adaptation restores named variables and their dimensions.

        Raises
        ------
        ValueError
            If both target groups or the observations are empty, or if target
            names overlap.
        """
        varying = [inference_variables] if isinstance(inference_variables, str) else list(inference_variables or [])
        invariant = [invariant_variables] if isinstance(invariant_variables, str) else list(invariant_variables or [])
        observations = [summary_variables] if isinstance(summary_variables, str) else list(summary_variables)
        if not (varying or invariant) or not observations:
            raise ValueError("At least one target group and observations must be nonempty.")
        if set(varying) & set(invariant):
            raise ValueError("Varying and invariant target names must not overlap.")
        adapter = Adapter().to_array().convert_dtype("float64", "float32")
        adapter.as_time_series(varying + observations)
        if varying:
            adapter.concatenate(varying, into="inference_variables")
        if invariant:
            adapter.concatenate(invariant, into="invariant_variables")
        adapter.concatenate(observations, into="summary_variables")
        keep = ["summary_variables"]
        if varying:
            keep.append("inference_variables")
        if invariant:
            keep.append("invariant_variables")
        if inference_conditions is not None:
            conditions = [inference_conditions] if isinstance(inference_conditions, str) else list(inference_conditions)
            adapter.concatenate(conditions, into="inference_conditions")
            keep.append("inference_conditions")
        for key, name in {
            "sample_weight": sample_weight,
            "summary_mask": summary_mask,
            "summary_attention_mask": summary_attention_mask,
            "inference_mask": inference_mask,
            "inference_attention_mask": inference_attention_mask,
        }.items():
            if name is not None:
                adapter.rename(name, key)
                keep.append(key)
        return adapter.keep(keep)

    def _validate_shapes(self, data_shapes, targets=True):
        """Check canonical batch, time, target, and mask dimensions.

        Parameters
        ----------
        data_shapes : mapping of str to tuple
            Shapes of adapted tensors. Observations must be (B, T, Dx), known
            conditions (B, C) or (B, T, C), and padding masks (B, T).
        targets : bool, optional, default: True
            Also require enabled varying targets shaped (B, T, Dv) and
            invariant targets shaped (B, Di).

        Raises
        ------
        ValueError
            If required arrays are absent, the observation sequence is empty,
            or batch and time dimensions are incompatible.
        """
        summary = data_shapes.get("summary_variables")
        if summary is None or len(summary) != 3:
            raise ValueError("summary_variables must have shape (B, T, Dx).")
        if summary[1] == 0:
            raise ValueError("The observation sequence must not be empty.")
        conditions = data_shapes.get("inference_conditions")
        if conditions is not None and (
            len(conditions) not in (2, 3)
            or conditions[0] != summary[0]
            or (len(conditions) == 3 and conditions[1] != summary[1])
        ):
            raise ValueError("inference_conditions must have shape (B, C) or (B, T, C).")
        for key in ("summary_mask", "inference_mask"):
            if data_shapes.get(key) is not None and tuple(data_shapes[key]) != tuple(summary[:2]):
                raise ValueError(f"{key} must have shape (B, T).")
        if targets:
            varying = data_shapes.get("inference_variables")
            invariant = data_shapes.get("invariant_variables")
            if self.has_varying and (varying is None or len(varying) != 3 or tuple(varying[:2]) != tuple(summary[:2])):
                raise ValueError("inference_variables must have shape (B, T, Dv), aligned with observations.")
            if self.has_invariant and (invariant is None or len(invariant) != 2 or invariant[0] != summary[0]):
                raise ValueError(
                    "invariant_variables must have shape (B, Di); tiled invariant targets are not supported."
                )

    def build(self, data_shapes):
        """Build the encoder, pooling layer, standardizers, and enabled heads.

        Parameters
        ----------
        data_shapes : mapping of str to tuple
            Shapes of adapted training arrays using the canonical keys.
            Both observations and all enabled target groups are required.

        Raises
        ------
        ValueError
            If input shapes are invalid, the summary network does not return
            (B, T, H), or invariant pooling does not return (B, G).
        """
        self._validate_shapes(data_shapes)
        self._data_shapes = dict(data_shapes)
        self.standardizer.build(data_shapes)
        summary_shape = tuple(data_shapes["summary_variables"])
        known_shape = data_shapes.get("inference_conditions")

        if known_shape is not None:
            summary_shape = (*summary_shape[:-1], summary_shape[-1] + known_shape[-1])
        if not self.summary_network.built:
            self.summary_network.build(summary_shape)
        feature_shape = tuple(self.summary_network.compute_output_shape(summary_shape))

        if len(feature_shape) != 3 or feature_shape[:2] != summary_shape[:2]:
            raise ValueError("summary_network must return one feature vector per observation: (B, T, H).")
        pooled_shape = (feature_shape[0], feature_shape[-1] + 1)

        if self.invariant_pooling is not None:
            if not self.invariant_pooling.built:
                self.invariant_pooling.build(feature_shape)
            pooled_shape = tuple(self.invariant_pooling.compute_output_shape(feature_shape))

        if len(pooled_shape) != 2 or pooled_shape[0] != summary_shape[0]:
            raise ValueError("invariant_pooling must map (B, T, H) to (B, G).")

        invariant_shape = tuple(data_shapes["invariant_variables"]) if self.has_invariant else None
        if self.has_invariant:
            self.invariant_approximator.build(
                {"inference_variables": invariant_shape, "inference_conditions": pooled_shape}
            )
        if self.has_varying:
            sequence_shapes = {"inference_variables": tuple(data_shapes["inference_variables"])}
            if self.joint:
                sequence_shapes["summary_variables"] = feature_shape
                if self.has_invariant:
                    sequence_shapes["inference_conditions"] = invariant_shape
            else:
                sequence_shapes["inference_conditions"] = (
                    *feature_shape[:-1],
                    feature_shape[-1] + (invariant_shape[-1] if self.has_invariant else 0),
                )
            self.sequence_approximator.build(sequence_shapes)
        self.built = True

    @staticmethod
    def _tensorize(data):
        """Convert leaves of a nested data structure to backend tensors.

        Parameters
        ----------
        data : nested structure of array-like
            Arrays to convert using the active Keras backend.

        Returns
        -------
        tensors : nested structure of tensors
            Converted arrays with the same structure as `data`.
        """
        return keras.tree.map_structure(keras.ops.convert_to_tensor, data)

    def _call_summary(self, inputs, mask=None, attention_mask=None, training=False):
        """Call the encoder with the mask arguments supported by its signature.

        Parameters
        ----------
        inputs : tensor
            Encoder inputs shaped (B, T, D), including any known conditions.
        mask : tensor or None, optional, default: None
            Observation padding mask shaped (B, T).
        attention_mask : tensor or None, optional, default: None
            Attention mask compatible with the encoder.
        training : bool, optional, default: False
            Whether the encoder runs in training mode.

        Returns
        -------
        features : tensor
            Encoded observations shaped (B, T, H).
        """
        kwargs = {"training": training}
        if mask is not None:
            kwargs["mask"] = mask
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        return self.summary_network(inputs, **filter_kwargs(kwargs, self.summary_network.call))

    def pool_invariants(self, features, mask=None, training=False):
        """Pool the complete encoded sequence for the invariant head.

        Parameters
        ----------
        features : tensor
            Encoded observations shaped (B, T, H).
        mask : tensor or None, optional, default: None
            Padding mask shaped (B, T), with True indicating valid steps.
            If None, all steps contribute.
        training : bool, optional, default: False
            Training flag forwarded to a custom pooling layer when supported.

        Returns
        -------
        pooled : tensor
            Invariant features shaped (B, G). Without a custom pooling layer,
            G = H + 1: the masked mean followed by log(1 + valid length).
            Fully masked sequences produce zeros under the default pooling.
        """
        if self.invariant_pooling is not None:
            kwargs = {"training": training}
            if mask is not None:
                kwargs["mask"] = mask
            return self.invariant_pooling(features, **filter_kwargs(kwargs, self.invariant_pooling.call))
        weights = keras.ops.ones_like(features[..., :1])
        if mask is not None:
            weights = keras.ops.cast(mask[..., None], features.dtype)
        count = keras.ops.sum(weights, axis=1)
        valid = keras.ops.where(keras.ops.cast(weights, "bool"), features, 0.0)
        mean = keras.ops.sum(valid, axis=1) / keras.ops.maximum(count, 1.0)
        return keras.ops.concatenate([mean, keras.ops.log1p(count)], axis=-1)

    def _encode(self, data, stage="inference"):
        """Standardize observations and compute sequence and invariant features.

        Parameters
        ----------
        data : mapping of str to tensor
            Canonical observations, optional known conditions, and masks.
        stage : str, optional, default: "inference"
            Stage forwarded to standardization. The encoder and pooling layer
            run in training mode only when this equals "training".

        Returns
        -------
        features : tensor
            Sequence features shaped (B, T, H). In filtering mode, causal
            encoders run once; other encoders are evaluated on prefixes.
        pooled : tensor or None
            Features shaped (B, G) pooled from the complete sequence, or None
            when the invariant head is disabled.
        """
        mask = data.get("summary_mask")
        inputs = self.standardizer.maybe_standardize(
            data["summary_variables"], key="summary_variables", stage=stage, mask=mask
        )
        known = self.standardizer.maybe_standardize(
            data.get("inference_conditions"),
            key="inference_conditions",
            stage=stage,
            mask=mask
            if data.get("inference_conditions") is not None and len(data["inference_conditions"].shape) == 3
            else None,
        )

        if known is not None:
            if len(known.shape) == 2:
                known = keras.ops.broadcast_to(known[:, None, :], (*keras.ops.shape(inputs)[:2], known.shape[-1]))
            inputs = keras.ops.concatenate([inputs, known], axis=-1)

        attention_mask = data.get("summary_attention_mask")
        features = self._call_summary(inputs, mask=mask, attention_mask=attention_mask, training=stage == "training")
        pooled = self.pool_invariants(features, mask=mask, training=stage == "training") if self.has_invariant else None

        # A causal encoder can process the sequence once. Other encoders must
        # see each prefix separately so varying factors cannot use future data.
        causal = (
            isinstance(self.summary_network, RecurrentNetwork) and not self.summary_network.bidirectional
        ) or getattr(self.summary_network, "supports_filtering", False)
        if self.mode == "filtering" and self.has_varying and not causal:
            prefix_features = []
            for stop in range(1, inputs.shape[1]):
                prefix = self._call_summary(
                    inputs[:, :stop],
                    mask=mask[:, :stop] if mask is not None else None,
                    attention_mask=attention_mask[:, :stop] if attention_mask is not None else None,
                    training=stage == "training",
                )
                prefix_features.append(prefix[:, -1:])
            features = keras.ops.concatenate([*prefix_features, features[:, -1:]], axis=1)

        return features, pooled

    def _sequence_data(self, data, features, invariant):
        """Arrange encoded observations and invariants for the varying head.

        Parameters
        ----------
        data : mapping of str to tensor
            Canonical data containing any available varying targets and masks.
        features : tensor
            Sequence features shaped (B, T, H).
        invariant : tensor or None
            Paired invariant targets or draws shaped (B, Di).

        Returns
        -------
        sequence_data : dict of str to tensor
            For autoregressive estimation, features become summary variables
            and invariants become conditions. For marginal estimation,
            invariants are broadcast across time and appended to features as
            conditions. Available target and attention masks are preserved.
        """
        result = {
            key: data[key]
            for key in ("inference_variables", "inference_mask", "inference_attention_mask", "summary_mask")
            if key in data
        }
        if self.joint:
            result["summary_variables"] = features
            if invariant is not None:
                result["inference_conditions"] = invariant
        elif invariant is not None:
            broadcast = keras.ops.broadcast_to(
                invariant[:, None, :], (*keras.ops.shape(features)[:2], invariant.shape[-1])
            )
            result["inference_conditions"] = keras.ops.concatenate([features, broadcast], axis=-1)
            result.pop("summary_mask", None)
        else:
            result["inference_conditions"] = features
            result.pop("summary_mask", None)

        return result

    def compute_metrics(
        self,
        summary_variables,
        inference_variables=None,
        invariant_variables=None,
        inference_conditions=None,
        sample_weight=None,
        summary_mask=None,
        summary_attention_mask=None,
        inference_mask=None,
        inference_attention_mask=None,
        stage="training",
    ):
        """Compute the enabled heads' metrics and combined training objective.

        Parameters
        ----------
        summary_variables : array-like
            Adapted observations shaped (B, T, Dx).
        inference_variables : array-like or None, optional, default: None
            Varying targets shaped (B, T, Dv). Required when the varying head
            is enabled.
        invariant_variables : array-like or None, optional, default: None
            Invariant targets shaped (B, Di). Required when the invariant head
            is enabled; tiled targets are not supported.
        inference_conditions : array-like or None, optional, default: None
            Known conditions shaped (B, C) or (B, T, C), appended to encoder
            inputs after standardization.
        sample_weight : array-like or None, optional, default: None
            Dataset weights shaped (B,). Broadcast over time for the varying
            head and multiplied by `inference_mask` when present.
        summary_mask : array-like or None, optional, default: None
            Observation padding mask shaped (B, T), with True indicating valid
            steps. Used for observation standardization, encoding, and pooling.
        summary_attention_mask : array-like or None, optional, default: None
            Attention mask forwarded to the summary network when supported.
        inference_mask : array-like or None, optional, default: None
            Target padding mask shaped (B, T), with True indicating steps
            included in the varying objective.
        inference_attention_mask : array-like or None, optional, default: None
            Attention mask forwarded to the varying approximator.
        stage : str, optional, default: "training"
            Stage passed to standardization and both child approximators,
            typically "training", "validation", or "inference".

        Returns
        -------
        metrics : dict of str to tensor
            Child metrics prefixed with "varying/" or "invariant/", plus the
            combined "loss" and layer-loss metrics. Child layer losses are
            removed before combining objectives and included once through the
            parent model's layer losses.

        Notes
        -----
        Inputs are already adapted; this method does not apply `adapter`.
        An unbuilt approximator is built from the supplied tensor shapes.
        Child objectives retain their native BayesFlow reductions. The varying
        training objective is generally averaged over time, whereas `log_prob`
        sums the sequence log densities.
        """
        data = self._tensorize(
            {
                key: value
                for key, value in {
                    "inference_variables": inference_variables,
                    "invariant_variables": invariant_variables,
                    "summary_variables": summary_variables,
                    "inference_conditions": inference_conditions,
                    "summary_mask": summary_mask,
                    "summary_attention_mask": summary_attention_mask,
                    "inference_mask": inference_mask,
                    "inference_attention_mask": inference_attention_mask,
                }.items()
                if value is not None
            }
        )
        if not self.built:
            self.build(keras.tree.map_structure(keras.ops.shape, data))

        features, pooled = self._encode(data, stage=stage)

        if sample_weight is not None:
            sample_weight = keras.ops.convert_to_tensor(sample_weight)

        invariant_metrics = (
            self.invariant_approximator.compute_metrics(
                inference_variables=data["invariant_variables"],
                inference_conditions=pooled,
                sample_weight=sample_weight,
                stage=stage,
            )
            if self.has_invariant
            else {}
        )

        sequence_weight = None if sample_weight is None else sample_weight[:, None]

        if inference_mask is not None:
            mask = keras.ops.cast(data["inference_mask"], features.dtype)
            sequence_weight = mask if sequence_weight is None else sequence_weight * mask

        sequence_metrics = (
            self.sequence_approximator.compute_metrics(
                **self._sequence_data(data, features, data.get("invariant_variables")),
                sample_weight=sequence_weight,
                stage=stage,
            )
            if self.has_varying
            else {}
        )

        metrics = {
            f"{name}/{key}": value
            for name, head in (("invariant", invariant_metrics), ("varying", sequence_metrics))
            for key, value in head.items()
        }

        loss = sum(head["loss"] - head.get("layer_loss", 0.0) for head in (invariant_metrics, sequence_metrics) if head)
        return metrics | self._with_layer_losses(loss)

    def _batch_size_from_data(self, data):
        """Read the batch size from canonical observations.

        Parameters
        ----------
        data : mapping of str to tensor
            Adapted data containing "summary_variables" shaped (B, T, Dx).

        Returns
        -------
        batch_size : int or scalar tensor
            Observation batch size B, regardless of the enabled target heads.
        """
        return keras.ops.shape(data["summary_variables"])[0]

    def fit(self, *args, **kwargs):
        """Train the approximator using its configured adapter.

        Parameters
        ----------
        *args
            Positional arguments forwarded to BayesFlow's `Approximator.fit`,
            subject to that method's signature.
        **kwargs
            Dataset or simulator configuration and training options forwarded
            to `Approximator.fit`. Any supplied adapter is replaced by this
            approximator's configured adapter.

        Returns
        -------
        history : keras.callbacks.History
            Training and validation losses and metrics recorded during fitting.
        """
        return super().fit(*args, **(kwargs | {"adapter": self.adapter}))

    def call(self, inputs=None, training=None, **kwargs):
        """Route Keras batch calls or explicit tensors to `compute_metrics`.

        Parameters
        ----------
        inputs : mapping, array-like, or None, optional, default: None
            Adapted batch mapping, or varying targets to use as
            "inference_variables". Other required tensors must be supplied
            through keyword arguments when `inputs` is not a mapping.
        training : bool or None, optional, default: None
            If given, select "training" or "validation" unless an explicit
            `stage` is already present in the keyword arguments.
        **kwargs
            Arguments for `compute_metrics`. For mapping inputs, these
            override matching batch entries and unsupported keys are filtered.

        Returns
        -------
        metrics : dict of str to tensor
            Combined loss and metrics returned by `compute_metrics`.
        """
        if training is not None:
            kwargs.setdefault("stage", "training" if training else "validation")
        if isinstance(inputs, Mapping):
            return self.compute_metrics(**filter_kwargs(dict(inputs) | kwargs, self.compute_metrics))
        if inputs is not None:
            kwargs["inference_variables"] = inputs
        return self.compute_metrics(**kwargs)

    def sample(
        self,
        num_samples: int,
        conditions: Mapping[str, np.ndarray],
        batch_size: int | None = None,
        split: bool = False,
        return_summaries: bool = False,
        seed=None,
        sequence_kwargs: Mapping | None = None,
        invariant_kwargs: Mapping | None = None,
    ) -> dict[str, np.ndarray]:
        """Sample paired invariant parameters and conditional trajectories.

        Parameters
        ----------
        num_samples : int
            Number S of paired posterior draws per dataset.
        conditions : mapping of str to np.ndarray
            Observations and optional known conditions and masks before
            adaptation. With an identity adapter, observations have shape
            (B, T, Dx) under "summary_variables". Target arrays are not required.
        batch_size : int or None, optional, default: None
            Number of datasets encoded and expanded at once. If None, process
            all datasets in one batch.
        split : bool, optional, default: False
            Split multichannel outputs along their final feature axis using
            BayesFlow's array splitting convention. Named scalar time series
            retain their time axis.
        return_summaries : bool, optional, default: False
            Include sequence features under "_summaries" with shape (B, T, H).
            When invariants are enabled, also include pooled features under
            "_invariant_summaries" with shape (B, G).
        seed : int, keras.random.SeedGenerator, or None, optional, default: None
            Random seed passed to the child samplers. If None, use this
            approximator's seed generator.
        sequence_kwargs : mapping or None, optional, default: None
            Additional arguments for the varying approximator's `sample`.
            Must not duplicate arguments supplied directly by this method,
            such as `num_samples`, `conditions`, `sample_shape`, or `seed`.
        invariant_kwargs : mapping or None, optional, default: None
            Additional arguments for the invariant approximator's `sample`,
            subject to the same restriction on duplicate arguments.

        Returns
        -------
        samples : dict of str to np.ndarray
            Samples after inverse adaptation. With an identity adapter,
            "inference_variables" has shape (B, S, T, Dv) and
            "invariant_variables" has shape (B, S, Di), for enabled heads.
            Named adapters restore original parameter keys and dimensions.
            Optional summaries have no posterior sample axis.

        Notes
        -----
        Each invariant draw conditions exactly one trajectory. Marginal mode
        samples per-time factors conditional on that draw; joint mode samples
        an autoregressive trajectory. Filtering factors use observation
        prefixes conditional on invariants inferred from the complete sequence.
        """

        adapted = self.adapter(conditions, strict=False, stage="inference")
        self._validate_shapes(keras.tree.map_structure(np.shape, adapted), targets=False)
        size = adapted["summary_variables"].shape[0]

        batch_size = size if batch_size is None else batch_size

        seed = resolve_seed(seed, self.seed_generator)

        batches = []
        for start in range(0, size, batch_size):
            batch = self._tensorize({key: value[start : start + batch_size] for key, value in adapted.items()})
            features, pooled = self._encode(batch)
            count, steps = keras.ops.shape(features)[:2]
            invariant = (
                self.invariant_approximator.sample(
                    num_samples=num_samples,
                    conditions={"inference_conditions": pooled},
                    sample_shape=(),
                    seed=seed,
                    **(invariant_kwargs or {}),
                )["inference_variables"]
                if self.has_invariant
                else None
            )
            # Expand only this batch. The native ancestral_sample implementation
            # uses generic continuous condition resolution even for its AR
            # subclass, so call each child's actual sample method explicitly.
            expanded = {
                key: keras.ops.repeat(value, num_samples, axis=0)
                for key, value in batch.items()
                if key not in {"inference_variables", "invariant_variables"}
            }
            expanded_features = keras.ops.repeat(features, num_samples, axis=0)
            flat_invariant = invariant.reshape(count * num_samples, -1) if invariant is not None else None
            sequence = (
                self.sequence_approximator.sample(
                    num_samples=1,
                    conditions=self._sequence_data(
                        expanded,
                        expanded_features,
                        self._tensorize(flat_invariant) if flat_invariant is not None else None,
                    ),
                    sample_shape=(steps,),
                    seed=seed,
                    **(sequence_kwargs or {}),
                )["inference_variables"][:, 0]
                if self.has_varying
                else None
            )

            # Invert on (B*S, T, D), so time-series adapter transforms see their
            # training axes, before introducing the posterior sample axis.
            varying_samples = (
                self.adapter({"inference_variables": sequence}, inverse=True, strict=False, stage="inference")
                if self.has_varying
                else {}
            )
            invariant_samples = (
                self.adapter({"invariant_variables": flat_invariant}, inverse=True, strict=False, stage="inference")
                if self.has_invariant
                else {}
            )
            if split:
                # Named scalar time series may already be (B*S, T). Their last
                # axis is time, whereas unsqueezed series still have channels.
                scalar_series = {key: value for key, value in varying_samples.items() if value.ndim == 2}
                channel_series = {key: value for key, value in varying_samples.items() if value.ndim != 2}
                varying_samples = scalar_series | split_arrays(channel_series, axis=-1)
                invariant_samples = split_arrays(invariant_samples, axis=-1)
            samples = varying_samples | invariant_samples
            samples = {key: value.reshape(count, num_samples, *value.shape[1:]) for key, value in samples.items()}

            if return_summaries:
                samples["_summaries"] = keras.ops.convert_to_numpy(features)
                if pooled is not None:
                    samples["_invariant_summaries"] = keras.ops.convert_to_numpy(pooled)
            batches.append(samples)

        return keras.tree.map_structure(lambda *values: np.concatenate(values, axis=0), *batches)

    def log_prob(self, data: Mapping[str, np.ndarray], return_components=False, **kwargs):
        """Evaluate the posterior log density with invariants counted once.

        Parameters
        ----------
        data : mapping of str to np.ndarray
            Targets, observations, and optional conditions and masks before
            adaptation. Canonical training shapes are (B, T, Dv) for varying
            targets, (B, Di) for invariants, and (B, T, Dx) for observations.
            There is no posterior sample axis.
        return_components : bool, optional, default: False
            Return separate "invariant" and "varying" log-density terms
            instead of their sum.
        **kwargs
            Additional arguments forwarded to both enabled child approximators'
            `log_prob` methods.

        Returns
        -------
        log_prob : np.ndarray or dict of str to np.ndarray
            Total log density shaped (B,), or the two component arrays, each
            shaped (B,), when `return_components` is True. Disabled heads
            contribute zero. Adapter target Jacobians are included.

        Raises
        ------
        ValueError
            If varying steps are masked and the adapter supplies a nonzero
            varying-target Jacobian. Dataset-level Jacobians cannot be
            apportioned across valid time steps.

        Notes
        -----
        The result is log q(phi | x) + log q(theta | phi, x). In marginal mode,
        the sequence term sums conditional per-time log densities; it does not
        model dependence between varying parameters beyond shared invariants.
        In joint mode, it is an autoregressive trajectory density. Filtering
        factors use observation prefixes conditional on phi.

        Target-changing adapters must provide their log-determinant Jacobians
        following BayesFlow's adapter contract. Densities are evaluated on the
        supplied parameter scale, without resolving simulator formulas or links.
        """
        adapted, jacobian = self.adapter(data, strict=False, log_det_jac=True, stage="inference")
        adapted = self._tensorize(adapted)

        if adapted.get("inference_mask") is not None:
            valid = keras.ops.convert_to_numpy(adapted["inference_mask"])
            if not np.all(valid) and np.any(np.asarray(jacobian.get("inference_variables", 0.0)) != 0):
                raise ValueError(
                    "Masked sequence log_prob requires zero varying-target adapter Jacobians. "
                    "BayesFlow's dataset-level Jacobian cannot be split over valid time points."
                )
        features, pooled = self._encode(adapted)
        invariant = (
            self.invariant_approximator.log_prob(
                {"inference_variables": adapted["invariant_variables"], "inference_conditions": pooled}, **kwargs
            )
            if self.has_invariant
            else np.zeros(features.shape[0])
        )
        sequence = (
            self.sequence_approximator.log_prob(
                self._sequence_data(adapted, features, adapted.get("invariant_variables")), **kwargs
            )
            if self.has_varying
            else np.zeros(features.shape[0])
        )
        if self.has_varying and not self.joint:
            if adapted.get("inference_mask") is not None:
                sequence = np.where(keras.ops.convert_to_numpy(adapted["inference_mask"]), sequence, 0.0)
            sequence = np.sum(sequence, axis=-1)
        invariant = invariant + jacobian.get("invariant_variables", 0.0)
        sequence = sequence + jacobian.get("inference_variables", 0.0)
        components = {"invariant": np.asarray(invariant), "varying": np.asarray(sequence)}
        return components if return_components else components["invariant"] + components["varying"]

    def summarize(self, conditions, **kwargs):
        """Encode observations into one summary vector per time step.

        Parameters
        ----------
        conditions : mapping of str to array-like
            Observations, optional known conditions, and masks before
            adaptation. Canonical observations have shape (B, T, Dx).
        **kwargs
            Accepted for compatibility with the BayesFlow summary interface;
            currently ignored.

        Returns
        -------
        summaries : np.ndarray
            Sequence features shaped (B, T, H), respecting the configured
            smoothing or filtering mode. Pooled invariant features are omitted.
        """
        adapted = self.adapter(conditions, strict=False, stage="inference")
        features, _ = self._encode(self._tensorize(adapted))
        return keras.ops.convert_to_numpy(features)

    def compile(self, *args, invariant_inference_metrics=None, **kwargs):
        """Compile training and optionally configure invariant-head metrics.

        Parameters
        ----------
        *args
            Positional arguments forwarded to `Approximator.compile`.
        invariant_inference_metrics : keras.metrics.Metric, sequence, or None, optional, default: None
            Metrics assigned to the invariant density network when that head
            is enabled. If None, retain its existing metrics.
        **kwargs
            Compilation options forwarded to BayesFlow and Keras, including
            optimizer settings, `inference_metrics` for the primary density
            network, and `summary_metrics` for the encoder. The primary density
            network is the varying head when enabled, otherwise the invariant
            head. Forwarded options are applied after the invariant metrics.
        """
        if invariant_inference_metrics is not None and self.has_invariant:
            self.invariant_inference_network._metrics = invariant_inference_metrics
        return super().compile(*args, **kwargs)

    def get_compile_config(self):
        """Return serialized compilation settings for checkpoint restoration.

        Returns
        -------
        config : dict
            Base compilation settings and invariant-head metrics when enabled,
            or an empty dictionary if the model has not been compiled.
        """
        if not self.compiled:
            return {}
        if not self.has_invariant:
            return super().get_compile_config()
        return super().get_compile_config() | serialize(
            {"invariant_inference_metrics": self.invariant_inference_network._metrics}
        )

    def get_config(self):
        """Serialize the network architecture, adapter, and head configuration.

        Returns
        -------
        config : dict
            Keras-compatible constructor configuration. Includes the decoder
            only for an enabled joint varying head. The `joint` flag is stored
            for this base class and inferred from the class for public subclasses.
        """
        config = {
            "varying_inference_network": self.varying_inference_network if self.has_varying else "coupling",
            "invariant_inference_network": self.invariant_inference_network if self.has_invariant else "coupling",
            "summary_network": self.summary_network,
            "invariant_pooling": self.invariant_pooling,
            "adapter": self.adapter,
            "mode": self.mode,
            "has_varying": self.has_varying,
            "has_invariant": self.has_invariant,
            "standardize": self._standardize,
        }
        if self.joint and self.has_varying:
            config["decoder_network"] = self.sequence_approximator.decoder_network
        if type(self) is CompositeApproximator:
            config["joint"] = self.joint
        return super().get_config() | serialize(config)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        """Restore an approximator from its serialized constructor configuration.

        Parameters
        ----------
        config : dict
            Configuration returned by `get_config`. Legacy configurations using
            "inference_network" are migrated to "varying_inference_network".
        custom_objects : dict or None, optional, default: None
            Custom classes or functions supplied to Keras deserialization.

        Returns
        -------
        approximator : CompositeApproximator
            Restored instance of the requested class. Network weights are
            restored separately by the checkpoint loading mechanism.
        """
        # Previously saved composite checkpoints used the asymmetric name.
        config = dict(config)
        if "inference_network" in config:
            config["varying_inference_network"] = config.pop("inference_network")
        return super().from_config(config, custom_objects=custom_objects)

    def get_build_config(self):
        """Return the canonical shapes needed to rebuild network weights.

        Returns
        -------
        config : dict
            Recorded training shapes under "data_shapes", or an empty
            dictionary when no build shapes have been recorded.
        """
        return {"data_shapes": self._data_shapes} if self._data_shapes is not None else {}

    def build_from_config(self, config):
        """Rebuild component networks from saved canonical data shapes.

        Parameters
        ----------
        config : dict
            Build configuration returned by `get_build_config`. If no
            "data_shapes" entry is present, leave the model unchanged.
        """
        if config.get("data_shapes") is not None:
            self.build(config["data_shapes"])
