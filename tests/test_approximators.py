"""Analytic factorization checks and integration with native BayesFlow networks."""

import keras
import numpy as np
import pytest

import bayesflow as bf
from bayesflow.utils import weighted_mean

import superstats as sup
from superstats.approximators import CompositeApproximator, JointApproximator, MarginalApproximator


@keras.saving.register_keras_serializable(package="superstats.tests")
class NormalHead(bf.networks.InferenceNetwork):
    """Known Gaussian conditionals make routing and densities independently checkable."""

    def __init__(self, sum_conditions=False, scale=1.0, offset_penalty=0.0, **kwargs):
        super().__init__(**kwargs)
        self.sum_conditions = sum_conditions
        self.scale = scale
        self.offset_penalty = offset_penalty

    def build(self, xz_shape, conditions_shape=None):
        self.target_dim = xz_shape[-1]
        self.offset = self.add_weight(
            shape=(self.target_dim,),
            initializer="zeros",
            name="offset",
            regularizer=keras.regularizers.L2(self.offset_penalty) if self.offset_penalty else None,
        )

    def mean(self, conditions):
        conditions = keras.ops.convert_to_tensor(conditions)
        mean = keras.ops.sum(conditions, axis=-1, keepdims=True) if self.sum_conditions else conditions[..., :1]
        return mean + self.offset

    def log_prob(self, samples, conditions=None, **kwargs):
        z = (keras.ops.convert_to_tensor(samples) - self.mean(conditions)) / self.scale
        return -0.5 * keras.ops.sum(z**2 + np.log(2 * np.pi * self.scale**2), axis=-1)

    def compute_metrics(self, x, conditions=None, sample_weight=None, stage="training", **kwargs):
        return {"loss": weighted_mean(-self.log_prob(x, conditions), sample_weight)}

    def sample(self, batch_shape, conditions=None, seed=None, **kwargs):
        noise = keras.random.normal((*batch_shape, self.target_dim), seed=seed)
        return self.mean(conditions) + self.scale * noise

    def get_config(self):
        return super().get_config() | {
            "sum_conditions": self.sum_conditions,
            "scale": self.scale,
            "offset_penalty": self.offset_penalty,
        }


@keras.saving.register_keras_serializable(package="superstats.tests")
class FullMeanEncoder(keras.layers.Layer):
    """Repeat the complete-sequence mean at every smoothing position."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, **kwargs):
        return keras.ops.broadcast_to(keras.ops.mean(inputs, axis=1, keepdims=True), keras.ops.shape(inputs))

    def compute_output_shape(self, input_shape):
        return input_shape


@keras.saving.register_keras_serializable(package="superstats.tests")
class ShiftDecoder(keras.layers.Layer):
    """A known Markov conditional, with matching teacher-forcing and cached paths."""

    supports_filtering = True

    def build(self, inference_variables_shape, encoder_outputs_shape):
        self.target_dim = inference_variables_shape[-1]

    def call(self, inference_variables, encoder_outputs, **kwargs):
        previous = keras.ops.concatenate(
            [keras.ops.zeros_like(inference_variables[:, :1]), inference_variables[:, :-1]], axis=1
        )
        return keras.ops.concatenate([previous, encoder_outputs], axis=-1)

    def compute_output_shape(self, inference_variables_shape, encoder_outputs_shape):
        return (*inference_variables_shape[:-1], self.target_dim + encoder_outputs_shape[-1])

    def initialize_cache(self, encoder_outputs, **kwargs):
        return {"encoder_outputs": encoder_outputs}

    def decode_step(self, previous_target, *, step, cache, **kwargs):
        features = cache["encoder_outputs"][:, step]
        if previous_target is None:
            previous_target = keras.ops.zeros((features.shape[0], self.target_dim))
        return keras.ops.concatenate([previous_target, features], axis=-1), cache


def batch(batch_size=2, steps=3, varying_dim=1, invariant_dim=1):
    return {
        "inference_variables": np.linspace(-0.8, 0.6, batch_size * steps * varying_dim, dtype="float32").reshape(
            batch_size, steps, varying_dim
        ),
        "invariant_variables": np.linspace(-0.3, 0.4, batch_size * invariant_dim, dtype="float32").reshape(
            batch_size, invariant_dim
        ),
        "summary_variables": np.arange(batch_size * steps, dtype="float32").reshape(batch_size, steps, 1) / 3,
    }


def analytic_approximator(cls, mode="smoothing", **kwargs):
    if cls is JointApproximator:
        kwargs.setdefault("decoder_network", ShiftDecoder())
    return cls(
        inference_network=NormalHead(sum_conditions=True),
        invariant_inference_network=NormalHead(),
        summary_network=FullMeanEncoder() if mode == "smoothing" else keras.layers.Identity(),
        mode=mode,
        standardize=None,
        **kwargs,
    )


@pytest.mark.parametrize("cls", [MarginalApproximator, JointApproximator])
@pytest.mark.parametrize("mode", ["smoothing", "filtering"])
def test_metrics_and_log_prob_match_known_conditional_factorization(cls, mode):
    data = batch()
    approximator = analytic_approximator(cls, mode)
    metrics = keras.tree.map_structure(keras.ops.convert_to_numpy, approximator.compute_metrics(**data))
    obs = data["summary_variables"]
    global_mean = obs.mean(axis=1)
    features = np.broadcast_to(global_mean[:, None], obs.shape)
    if mode == "filtering":
        features = obs
    sequence_mean = features + data["invariant_variables"][:, None]
    if cls is JointApproximator:
        sequence_mean += np.concatenate(
            [np.zeros_like(data["inference_variables"][:, :1]), data["inference_variables"][:, :-1]], axis=1
        )
    constant = np.log(2 * np.pi)
    invariant_lp = -0.5 * ((data["invariant_variables"] - global_mean) ** 2 + constant).sum(axis=-1)
    step_lp = -0.5 * ((data["inference_variables"] - sequence_mean) ** 2 + constant).sum(axis=-1)
    np.testing.assert_allclose(metrics["invariant/loss"], -invariant_lp.mean(), rtol=1e-6)
    np.testing.assert_allclose(metrics["varying/loss"], -step_lp.mean(), rtol=1e-6)
    np.testing.assert_allclose(metrics["loss"], -invariant_lp.mean() - step_lp.mean(), rtol=1e-6)
    components = approximator.log_prob(data, return_components=True)
    np.testing.assert_allclose(components["invariant"], invariant_lp, rtol=1e-6)
    np.testing.assert_allclose(components["varying"], step_lp.sum(axis=-1), rtol=1e-6)
    np.testing.assert_allclose(approximator.log_prob(data), invariant_lp + step_lp.sum(axis=-1), rtol=1e-6)


@pytest.mark.parametrize("cls", [MarginalApproximator, JointApproximator])
@pytest.mark.parametrize("mode", ["smoothing", "filtering"])
def test_sampling_preserves_invariant_pairs_and_autoregressive_history(cls, mode):
    data = batch(batch_size=3)
    approximator = analytic_approximator(cls, mode)
    approximator.compute_metrics(**data)
    # Essentially deterministic sequence conditionals let us check every pairing.
    approximator.inference_network.scale = 1e-6
    conditions = {"summary_variables": data["summary_variables"]}
    result = approximator.sample(num_samples=5, conditions=conditions, seed=42, batch_size=2)
    assert result["invariant_variables"].shape == (3, 5, 1)
    assert result["inference_variables"].shape == (3, 5, 3, 1)
    obs = conditions["summary_variables"]
    features = np.broadcast_to(obs.mean(axis=1, keepdims=True), obs.shape)
    if mode == "filtering":
        features = obs
    expected = features[:, None] + result["invariant_variables"][:, :, None]
    if cls is JointApproximator:
        expected = expected.cumsum(axis=2)
    np.testing.assert_allclose(result["inference_variables"], expected, atol=1e-5)
    again = approximator.sample(num_samples=5, conditions=conditions, seed=42, batch_size=2)
    for key in result:
        np.testing.assert_array_equal(result[key], again[key])


@pytest.mark.parametrize("cls", [MarginalApproximator, JointApproximator])
def test_filtering_sequence_features_do_not_see_future_but_global_head_does(cls):
    data = batch()
    approximator = analytic_approximator(cls, "filtering")
    approximator.compute_metrics(**data)
    changed = {key: value.copy() for key, value in data.items()}
    changed["summary_variables"][:, -1] += 100
    original_features = approximator.summarize(data)
    changed_features = approximator.summarize(changed)
    np.testing.assert_array_equal(original_features[:, :-1], changed_features[:, :-1])
    assert not np.allclose(original_features[:, -1], changed_features[:, -1])
    assert not np.allclose(
        approximator.log_prob(data, return_components=True)["invariant"],
        approximator.log_prob(changed, return_components=True)["invariant"],
    )


@pytest.mark.parametrize("cls", [MarginalApproximator, JointApproximator])
def test_masks_and_batch_weights(cls):
    data = batch()
    data["summary_mask"] = np.array([[True, True, False], [True, True, False]])
    data["inference_mask"] = data["summary_mask"]
    approximator = analytic_approximator(cls)
    sample_weight = np.array([1.0, 2.0])
    original = approximator.compute_metrics(**data, sample_weight=sample_weight)
    invariant_lp = -0.5 * (
        (data["invariant_variables"] - data["summary_variables"].mean(axis=1)) ** 2 + np.log(2 * np.pi)
    ).sum(axis=-1)
    expected_invariant_loss = -(invariant_lp[0] + 2 * invariant_lp[1]) / 2
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(original["invariant/loss"]), expected_invariant_loss, rtol=1e-6
    )
    changed = {key: value.copy() for key, value in data.items()}
    changed["inference_variables"][:, -1] = 1000
    after = approximator.compute_metrics(**changed, sample_weight=sample_weight)
    np.testing.assert_allclose(keras.ops.convert_to_numpy(original["loss"]), keras.ops.convert_to_numpy(after["loss"]))
    np.testing.assert_allclose(approximator.log_prob(data), approximator.log_prob(changed))


@pytest.mark.parametrize("cls", [MarginalApproximator, JointApproximator])
@pytest.mark.parametrize("steps", [1, 3])
def test_named_adapter_samples_restore_keys_and_axes(cls, steps):
    adapter = cls.build_adapter(
        inference_variables=["a", "b"], invariant_variables=["eta", "sigma"], summary_variables="x"
    )
    raw = {
        "a": np.ones((2, steps), dtype="float32"),
        "b": np.ones((2, steps, 2), dtype="float32"),
        "eta": np.ones((2, 1), dtype="float32"),
        "sigma": np.ones((2, 2), dtype="float32"),
        "x": np.ones((2, steps), dtype="float32"),
    }
    approximator = analytic_approximator(cls, adapter=adapter)
    approximator.compute_metrics(**adapter(raw))
    result = approximator.sample(num_samples=4, conditions={"x": raw["x"]}, seed=1)
    assert set(result) == {"a", "b", "eta", "sigma"}
    assert result["a"].shape == (2, 4, steps)
    assert result["b"].shape == (2, 4, steps, 2)
    assert result["eta"].shape == (2, 4, 1)
    assert result["sigma"].shape == (2, 4, 2)
    assert approximator.log_prob(raw).shape == (2,)
    split = approximator.sample(
        num_samples=4, conditions={"x": raw["x"]}, seed=1, split=True, return_summaries=True, batch_size=1
    )
    assert split["a"].shape == (2, 4, steps)
    assert split["b_0"].shape == (2, 4, steps)
    assert split["b_1"].shape == (2, 4, steps)
    assert split["eta"].shape == (2, 4)
    assert split["sigma_0"].shape == (2, 4)
    assert split["_summaries"].shape == (2, steps, 1)
    assert split["_invariant_summaries"].shape == (2, 2)


@pytest.mark.parametrize("cls", [MarginalApproximator, JointApproximator])
def test_adapter_target_jacobians_count_each_invariant_once(cls):
    data = batch()
    adapter = bf.Adapter().scale("inference_variables", 2.0).scale("invariant_variables", 3.0)
    transformed = adapter(data)
    approximator = analytic_approximator(cls, adapter=adapter)
    reference = analytic_approximator(cls)
    approximator.compute_metrics(**transformed)
    reference.compute_metrics(**transformed)
    expected = reference.log_prob(transformed) + 3 * np.log(2.0) + np.log(3.0)
    np.testing.assert_allclose(approximator.log_prob(data), expected, rtol=1e-6)


@pytest.mark.parametrize("cls", [MarginalApproximator, JointApproximator])
@pytest.mark.parametrize("known_time_series", [False, True])
def test_known_conditions_enter_shared_encoder(cls, known_time_series):
    data = batch()
    data["inference_conditions"] = np.ones((2, 3, 2) if known_time_series else (2, 2), dtype="float32")
    approximator = analytic_approximator(cls)
    assert np.isfinite(keras.ops.convert_to_numpy(approximator.compute_metrics(**data)["loss"]))
    result = approximator.sample(
        num_samples=2,
        conditions={key: value for key, value in data.items() if key in {"summary_variables", "inference_conditions"}},
        seed=2,
    )
    assert result["inference_variables"].shape == (2, 2, 3, 1)
    assert np.isfinite(approximator.log_prob(data)).all()


@pytest.mark.parametrize("cls", [MarginalApproximator, JointApproximator])
@pytest.mark.parametrize("mode", ["smoothing", "filtering"])
def test_native_component_configuration_round_trip(cls, mode):
    kwargs = {}
    if cls is JointApproximator:
        kwargs["decoder_network"] = (
            bf.networks.decoders.RecurrentDecoder(embed_dim=4)
            if mode == "filtering"
            else bf.networks.decoders.TransformerDecoder(embed_dim=4, num_heads=1, num_layers=1, dropout=0)
        )
    approximator = cls(
        inference_network=bf.networks.CouplingFlow(depth=1, subnet_kwargs={"widths": (8,)}),
        invariant_inference_network=bf.networks.CouplingFlow(depth=1, subnet_kwargs={"widths": (8,)}),
        summary_network=bf.networks.RecurrentNetwork(
            summary_dim=4, hidden_dim=4, bidirectional=mode == "smoothing", return_sequences=True, dropout=0
        ),
        mode=mode,
        **kwargs,
    )

    config = keras.saving.serialize_keras_object(approximator)
    restored = keras.saving.deserialize_keras_object(config)

    assert isinstance(restored, cls)
    assert restored.mode == mode
    if cls is JointApproximator:
        assert isinstance(restored.sequence_approximator.encoder_network, keras.layers.Identity)
        assert isinstance(restored.decoder_network, type(approximator.decoder_network))


@pytest.mark.parametrize("cls", [MarginalApproximator, JointApproximator])
def test_training_updates_both_heads_and_shared_encoder(cls, tmp_path):
    data = batch()
    data["invariant_variables"] += 2
    approximator = cls(
        inference_network=NormalHead(sum_conditions=True),
        invariant_inference_network=NormalHead(),
        summary_network=keras.layers.Dense(2),
        standardize=None,
        **({"decoder_network": ShiftDecoder()} if cls is JointApproximator else {}),
    )
    approximator.compute_metrics(**data)
    before = [np.array(keras.ops.convert_to_numpy(value)) for value in approximator.trainable_variables]
    approximator.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01))
    dataset = bf.datasets.OfflineDataset(data=data, batch_size=2, adapter=approximator.adapter)
    history = approximator.fit(dataset=dataset, epochs=1, verbose=0)
    assert np.isfinite(history.history["loss"]).all()
    after = [keras.ops.convert_to_numpy(value) for value in approximator.trainable_variables]
    assert all(not np.array_equal(old, new) for old, new in zip(before, after))
    path = tmp_path / "trained.keras"
    approximator.save(path)
    restored = keras.saving.load_model(path)
    np.testing.assert_allclose(restored.log_prob(data), approximator.log_prob(data), rtol=1e-6)
    assert int(keras.ops.convert_to_numpy(restored.optimizer.iterations)) == 1
    restored.fit(dataset=dataset, epochs=1, verbose=0)
    assert int(keras.ops.convert_to_numpy(restored.optimizer.iterations)) == 2


@pytest.mark.parametrize("cls", [MarginalApproximator, JointApproximator])
def test_shared_and_child_regularizers_are_added_once(cls):
    data = batch()
    approximator = cls(
        inference_network=NormalHead(sum_conditions=True, offset_penalty=0.1),
        invariant_inference_network=NormalHead(offset_penalty=0.1),
        summary_network=keras.layers.Dense(2, kernel_regularizer=keras.regularizers.L2(0.1)),
        standardize=None,
        **({"decoder_network": ShiftDecoder()} if cls is JointApproximator else {}),
    )
    approximator.compute_metrics(**data)
    approximator.inference_network.offset.assign(np.ones((1,), dtype="float32"))
    approximator.invariant_inference_network.offset.assign(np.ones((1,), dtype="float32"))
    metrics = approximator.compute_metrics(**data)
    components = approximator.log_prob(data, return_components=True)
    kernel = keras.ops.convert_to_numpy(approximator.summary_network.kernel)
    penalty = 0.1 * (kernel**2).sum() + 0.2
    expected = -components["invariant"].mean() - components["varying"].mean() / 3 + penalty
    np.testing.assert_allclose(keras.ops.convert_to_numpy(metrics["loss"]), expected, rtol=1e-6)


@pytest.mark.parametrize("cls", [MarginalApproximator, JointApproximator])
def test_masked_nonzero_adapter_jacobian_is_rejected(cls):
    data = batch()
    data["inference_mask"] = np.array([[True, True, False], [True, True, False]])
    approximator = analytic_approximator(cls, adapter=bf.Adapter().scale("inference_variables", 2.0))
    approximator.compute_metrics(**approximator.adapter(data))
    with pytest.raises(ValueError, match="dataset-level Jacobian"):
        approximator.log_prob(data)


def test_invalid_shapes_modes_and_decoder_fail_clearly():
    with pytest.raises(ValueError, match="mode"):
        analytic_approximator(MarginalApproximator, "unknown")
    with pytest.raises(ValueError, match="unrestricted cross-attention"):
        analytic_approximator(JointApproximator, "filtering", decoder_network=bf.networks.decoders.TransformerDecoder())
    with pytest.raises(ValueError, match="one feature vector"):
        MarginalApproximator(
            inference_network=NormalHead(),
            invariant_inference_network=NormalHead(),
            summary_network=keras.layers.GlobalAveragePooling1D(),
        ).compute_metrics(**batch())


def test_public_imports_and_internal_component_types():
    assert sup.CompositeApproximator is CompositeApproximator
    assert sup.MarginalApproximator is MarginalApproximator
    assert sup.JointApproximator is JointApproximator
    marginal = analytic_approximator(MarginalApproximator)
    joint = analytic_approximator(JointApproximator)
    assert isinstance(marginal, CompositeApproximator)
    assert isinstance(joint, CompositeApproximator)
    assert marginal.invariant_pooling is None
    assert type(marginal.sequence_approximator) is bf.approximators.ContinuousApproximator
    assert type(joint.sequence_approximator) is bf.approximators.AutoregressiveApproximator
    assert type(joint.invariant_approximator) is bf.approximators.ContinuousApproximator


def test_default_pooling_excludes_padding_without_a_layer():
    approximator = analytic_approximator(MarginalApproximator)
    features = np.array([[[2.0, 4.0], [4.0, 8.0], [np.nan, np.nan]]], dtype="float32")
    mask = np.array([[True, True, False]])
    pooled = keras.ops.convert_to_numpy(approximator.pool_invariants(keras.ops.convert_to_tensor(features), mask))
    np.testing.assert_allclose(pooled, [[3.0, 6.0, np.log(3.0)]], rtol=1e-6)
    assert approximator.invariant_pooling is None


@pytest.mark.parametrize("cls", [MarginalApproximator, JointApproximator])
def test_positive_invariants_use_adapted_conditions_and_one_log_jacobian(cls):
    latent = batch()
    physical = latent | {"invariant_variables": np.exp(latent["invariant_variables"])}
    reference = analytic_approximator(cls)
    approximator = analytic_approximator(cls, adapter=bf.Adapter().log("invariant_variables"))
    reference.compute_metrics(**latent)
    approximator.compute_metrics(**approximator.adapter(physical))
    np.testing.assert_allclose(
        approximator.log_prob(physical),
        reference.log_prob(latent) - latent["invariant_variables"].sum(axis=-1),
        rtol=1e-6,
    )
    conditions = {"summary_variables": latent["summary_variables"]}
    expected = reference.sample(num_samples=4, conditions=conditions, seed=17)
    actual = approximator.sample(num_samples=4, conditions=conditions, seed=17)
    assert (actual["invariant_variables"] > 0).all()
    np.testing.assert_allclose(actual["invariant_variables"], np.exp(expected["invariant_variables"]), rtol=1e-6)
    np.testing.assert_allclose(actual["inference_variables"], expected["inference_variables"], rtol=1e-6)


@pytest.mark.parametrize("joint", [False, True])
def test_public_base_can_be_saved_and_restored(tmp_path, joint):
    approximator = CompositeApproximator(
        inference_network=NormalHead(),
        invariant_inference_network=NormalHead(),
        summary_network=FullMeanEncoder(),
        decoder_network=ShiftDecoder() if joint else None,
        joint=joint,
        standardize=None,
    )
    approximator.compute_metrics(**batch())
    path = tmp_path / "base.keras"
    approximator.save(path)
    restored = keras.saving.load_model(path)
    assert type(restored) is CompositeApproximator
    assert restored.joint is joint
    assert restored.invariant_pooling is None
    np.testing.assert_allclose(restored.log_prob(batch()), approximator.log_prob(batch()), rtol=1e-6)
