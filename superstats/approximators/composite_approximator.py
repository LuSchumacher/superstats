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
    """Implementation shared by the marginal and autoregressive public APIs.

    Enabled targets use two canonical keys: ``inference_variables`` (B, T, Dv) and
    ``invariant_variables`` (B, Di). Observations use ``summary_variables``
    (B, T, Dx). Invariants are broadcast only as sequence-head conditions.
    Either target group can be disabled with has_varying/has_invariant.

    Filtering is conditional on the invariant parameters. The global head
    always sees the complete observation sequence, including in filtering mode.
    Consequently, ancestral samples with inferred invariants are not strict
    online filtering samples. Strict filtering requires applying the complete
    approximator separately to each observation prefix.
    """

    def __init__(
        self,
        *,
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
        """Primary density network required by BayesFlow's workflow interface."""
        return self.varying_inference_network or self.invariant_inference_network

    @property
    def varying_inference_network(self):
        return self.sequence_approximator.inference_network if self.has_varying else None

    @property
    def invariant_inference_network(self):
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
        """Build an adapter from named, separate varying and invariant targets.

        Pass None for an absent target group and disable that head in the constructor.
        Varying targets and observations may be (B, T) or (B, T, D).
        Invariant targets must be (B, D), including D=1. Known conditions
        must be either (B, C) or (B, T, C). Target groups must not overlap.
        Sample weights must have shape (B,).
        Inverse adaptation preserves the named variables' usual dimensions.
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
        return keras.tree.map_structure(keras.ops.convert_to_tensor, data)

    def _call_summary(self, inputs, *, mask=None, attention_mask=None, training=False):
        kwargs = {"training": training}
        if mask is not None:
            kwargs["mask"] = mask
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        return self.summary_network(inputs, **filter_kwargs(kwargs, self.summary_network.call))

    def pool_invariants(self, features, mask=None, training=False):
        """Pool the complete trajectory for the invariant head.

        The default is a plain masked mean with log(1 + valid length)
        appended. A learned pooling layer can be supplied explicitly.
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
        """Route adapted tensors to both children and sum their native objectives.
        ``inference_mask`` excludes padded sequence targets from the objective.
        Metrics are prefixed with ``varying/`` and ``invariant/``. Native head
        losses retain BayesFlow's reductions; the sequence objective is usually
        averaged over time, while ``log_prob`` sums densities over time.
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
        """Both target configurations have an observation batch axis."""
        return keras.ops.shape(data["summary_variables"])[0]

    def fit(self, *args, **kwargs):
        return super().fit(*args, **(kwargs | {"adapter": self.adapter}))

    def call(self, inputs=None, training=None, **kwargs):
        """Support Keras' dictionary-batch calls as well as explicit target calls."""
        if training is not None:
            kwargs.setdefault("stage", "training" if training else "validation")
        if isinstance(inputs, Mapping):
            return self.compute_metrics(**filter_kwargs(dict(inputs) | kwargs, self.compute_metrics))
        if inputs is not None:
            kwargs["inference_variables"] = inputs
        return self.compute_metrics(**kwargs)

    def sample(
        self,
        *,
        num_samples: int,
        conditions: Mapping[str, np.ndarray],
        batch_size: int | None = None,
        split: bool = False,
        return_summaries: bool = False,
        seed=None,
        sequence_kwargs: Mapping | None = None,
        invariant_kwargs: Mapping | None = None,
    ) -> dict[str, np.ndarray]:
        """Draw one trajectory per invariant draw, preserving paired sample axes.

        With an identity adapter, returns ``inference_variables`` (B, S, T, Dv)
        and ``invariant_variables`` (B, S, Di). Named adapters restore original
        parameter keys. ``batch_size`` bounds the number of datasets encoded and
        expanded at once. Child sampling options use separate kwargs mappings.
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

    def log_prob(self, data: Mapping[str, np.ndarray], *, return_components=False, **kwargs):
        """Evaluate log q(phi|x) + log q(theta|phi,x), with invariants counted once.

        Inputs have training shapes (B, T, Dv), (B, Di), (B, T, Dx).
        Returns (B,). ``return_components=True`` instead returns the two (B,)
        terms under ``invariant`` and ``varying``. For marginal estimation the
        sequence term is a product of conditional per-time marginals, not an
        estimate of the dependent joint trajectory posterior. For filtering,
        its factors use observation prefixes conditional on phi.

        Target-changing adapters must supply their Jacobian, following BayesFlow.
        Masked sequence densities require zero varying-target adapter Jacobians,
        since BayesFlow adapters return a single Jacobian per complete dataset.
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
        adapted = self.adapter(conditions, strict=False, stage="inference")
        features, _ = self._encode(self._tensorize(adapted))
        return keras.ops.convert_to_numpy(features)

    def compile(self, *args, invariant_inference_metrics=None, **kwargs):
        if invariant_inference_metrics is not None and self.has_invariant:
            self.invariant_inference_network._metrics = invariant_inference_metrics
        return super().compile(*args, **kwargs)

    def get_compile_config(self):
        if not self.compiled:
            return {}
        if not self.has_invariant:
            return super().get_compile_config()
        return super().get_compile_config() | serialize(
            {"invariant_inference_metrics": self.invariant_inference_network._metrics}
        )

    def get_config(self):
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
        # Previously saved composite checkpoints used the asymmetric name.
        config = dict(config)
        if "inference_network" in config:
            config["varying_inference_network"] = config.pop("inference_network")
        return super().from_config(config, custom_objects=custom_objects)

    def get_build_config(self):
        return {"data_shapes": self._data_shapes} if self._data_shapes is not None else {}

    def build_from_config(self, config):
        if config.get("data_shapes") is not None:
            self.build(config["data_shapes"])
