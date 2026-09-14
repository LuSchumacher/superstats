# Posterior approximators

Superstats provides two BayesFlow approximators sharing the public
`CompositeApproximator` base for unknown varying
and invariant parameters. Both own a shared observation encoder and construct
two posterior components internally:

| Class | Sequence component | Invariant component |
|---|---|---|
| `MarginalApproximator` | `ContinuousApproximator` | `ContinuousApproximator` |
| `JointApproximator` | `AutoregressiveApproximator` | `ContinuousApproximator` |

Use `mode="smoothing"` or `mode="filtering"` with either class. Smoothing state
factors use all observations. Filtering state factors use observation prefixes.
The invariant head always uses the complete sequence.

## Construct an approximator

```python
import bayesflow as bf
import superstats as sup
from superstats.approximators import JointApproximator, MarginalApproximator

approximator = JointApproximator(
    summary_network=bf.networks.RecurrentNetwork(
        return_sequences=True, bidirectional=True, summary_dim=64
    ),
    inference_network=bf.networks.CouplingFlow(),
    invariant_inference_network=bf.networks.CouplingFlow(),
    decoder_network=bf.networks.decoders.TransformerDecoder(),
    mode="smoothing",
)
```

The marginal class has the same interface without `decoder_network`. The
inference networks must be separate instances. Omitting `summary_network`
constructs a bidirectional recurrent encoder for smoothing or a unidirectional
recurrent encoder for filtering. Omitting the joint decoder constructs a
`TransformerDecoder` for smoothing or a `RecurrentDecoder` for filtering.

`summary_network` must preserve the time dimension: `(B, T, Dx) -> (B, T, H)`.
Known `inference_conditions`, if provided, are broadcast when necessary and
concatenated to the observations **before** encoding, increasing its input
feature dimension. The unknown invariant parameters are attached **after**
encoding, only for the conditional sequence head.

The sequence head receives invariant values in the adapter's coordinates.
For example, with `.log("invariant_variables")` for positive invariants,
training and ancestral sampling condition the sequence head on log parameters.
Returned samples are transformed back to physical coordinates, and `log_prob`
includes the invariant transformation Jacobian once per dataset.

By default, the invariant head receives the masked mean of the full observation
encoding, with `log(1 + valid sequence length)` appended. This is a simple
`CompositeApproximator.pool_invariants` call, with no parameterless Keras layer.
To use learned pooling,
pass `invariant_pooling`, a Keras layer mapping `(B, T, H) -> (B, G)`. It should
accept `mask` when using padded sequences. The encoder and pooling layer are
owned once by the composite, so they are shared by both objectives and saved
once in Keras checkpoints.

## Shapes and adapters

Use separate target tensors. Invariant targets have no time dimension; tiled
invariant targets are rejected. `B` is the number of datasets, `T` the number of
time points, and `S` the number of posterior samples.

| Canonical key | Training / `log_prob` | Sampling input | Sampling output |
|---|---|---|---|
| `inference_variables` | `(B, T, Dv)` | Omitted | `(B, S, T, Dv)` |
| `invariant_variables` | `(B, Di)` | Omitted | `(B, S, Di)` |
| `summary_variables` | `(B, T, Dx)` | `(B, T, Dx)` | Omitted |
| `inference_conditions` | `(B, C)` or `(B, T, C)` | Same | Omitted |
| `summary_mask` | `(B, T)` | Same | Omitted |
| `inference_mask` | `(B, T)` | Same | Omitted |

These shapes apply with the default identity adapter. To work with simulator
parameter names, use the class's `build_adapter`:

```python
adapter = JointApproximator.build_adapter(
    inference_variables=["v", "a"],
    invariant_variables=["v_sigma", "t0"],
    summary_variables=["time_steps", "rt", "choice"],
)

approximator = JointApproximator(
    inference_network=bf.networks.CouplingFlow(),
    invariant_inference_network=bf.networks.CouplingFlow(),
    adapter=adapter,
)
```

Named observations and varying targets can be `(B, T)` or `(B, T, D)`. Named
invariant targets must be `(B, D)`. Samples restore the original parameter names;
singleton varying channels are squeezed by the time-series adapter, giving
`(B, S, T)`, while invariant outputs retain `(B, S, D)`. Inverse adaptation runs
before adding the sample axis, including for sequences with `T=1`.

For simulations from `Model`, use `tile_to_steps=False` (the default).
`Workflow(model=model)` now constructs a marginal smoother with separate heads
when the model has both local and invariant parameters. Its default adapter
preserves named varying channels, returning `(B, S, T, D)` varying samples and
`(B, S, D)` invariant samples. Online training also leaves invariants untiled.

```python
workflow = sup.Workflow(
    model=model,
    embedding_network=bf.networks.RecurrentNetwork(return_sequences=True),
    inference_network=bf.networks.CouplingFlow(),
    invariant_inference_network=bf.networks.CouplingFlow(),
)
```

For joint smoothing or filtering, construct a `JointApproximator` with
`Workflow.default_adapter(model)` and pass it as
`Workflow(model=model, approximator=approximator)`. The workflow uses its networks
and adapter directly. Models without both target groups retain the ordinary
continuous approximator. For an invariant-only model, the default recurrent
encoder returns one global summary, so its targets also remain untiled.

## Metrics, sampling, and density evaluation

```python
training_batch = adapter(model.sample(batch_size=32, num_steps=100, tile_to_steps=False))
metrics = approximator.compute_metrics(**training_batch)

# Train both child heads and the shared encoder together.
approximator.compile(optimizer="adam")
dataset = bf.datasets.OfflineDataset(
    data=simulations, adapter=adapter, batch_size=32
)
approximator.fit(dataset=dataset, epochs=10)

samples = approximator.sample(
    conditions=observations, num_samples=500, batch_size=4, seed=123
)
log_density = approximator.log_prob(simulations)  # shape (B,)
components = approximator.log_prob(simulations, return_components=True)
# components["invariant"] and components["varying"] each have shape (B,).
```

`compute_metrics` receives **adapted** tensors, routes the true invariant values
to the conditional sequence head, and returns native head metrics prefixed with
`invariant/` and `varying/`, plus the combined `loss`. It sums the children's
native objectives and adds layer regularizers once. BayesFlow flow objectives
typically average sequence losses over time; density evaluation sums them.
`sample_weight` weights sequence targets and accepts `(B,)` or `(B, T)`.
`invariant_sample_weight` independently weights the global objective and accepts
`(B,)`. `inference_mask` also excludes invalid sequence targets from the loss and
sequence log density. Masks use `True` for valid points; an augmentation mask
where `True` means missing must be converted to the desired validity mask.

Sampling first draws `S` invariant vectors, then draws one sequence conditional
on each vector. The resulting sample axes remain paired; never independently
shuffle the two output groups. `batch_size` limits the number of datasets
encoded and expanded at once. For child-specific generation options, pass
`sequence_kwargs={...}` or `invariant_kwargs={...}`. `return_summaries=True`
also returns `_summaries` `(B, T, H)` and `_invariant_summaries` `(B, G)`.

`standardize="all"` is the default. Pass `None` to disable standardization, or
a subset of `inference_variables`, `invariant_variables`, `summary_variables`,
and `inference_conditions`. Targets are standardized independently in their
respective child approximators. Observation standardization happens before
shared encoding. `log_prob` includes target standardization and adapter
Jacobians, counting the invariant term once. Evaluate one target vector and
trajectory per dataset; to evaluate many posterior draws, broadcast observations
and flatten dataset and draw axes before calling `log_prob`.

With an `inference_mask` excluding time points, `log_prob` accepts varying-target
adapters with zero Jacobians. It rejects nonzero varying-target adapter
Jacobians in that case: BayesFlow reports one aggregate Jacobian per dataset,
which cannot be apportioned to the valid time points. Target standardization
inside the child approximators remains supported with masks.

The classes support `approximator.save("model.keras")` and
`keras.saving.load_model("model.keras")`. Custom networks and pooling layers
must be Keras serializable.

## Distribution assumptions

The marginal smoothing approximation samples from

$$
q(\phi\mid x_{1:T})\prod_t q(\theta_t\mid\phi,x_{1:T}).
$$

The per-time targets are independent conditional on observations and invariants.
Sampling a shared invariant vector induces dependence after marginalizing it,
but does not recover other posterior trajectory dependencies. Its `log_prob`
evaluates this product approximation, not the dependent joint posterior and
not an invariant-marginalized per-time density.

The joint smoothing approximation samples from

$$
q(\phi\mid x_{1:T})\prod_t
q(\theta_t\mid\theta_{<t},\phi,x_{1:T}).
$$

Training supplies shifted true sequence targets to the decoder. Sampling uses
BayesFlow's cached autoregressive decoder. This chain-rule factorization does
not impose conditional independence between time points.

Filtering replaces the observations in each sequence factor with `x_1:t`.
For arbitrary encoders, the composite evaluates each observation prefix and
keeps its final encoded vector. A known unidirectional BayesFlow recurrent
encoder uses one pass instead. Joint filtering accepts `RecurrentDecoder`, or
a custom decoder declaring `supports_filtering=True` and guaranteeing no
attention to future encoded positions. The native `TransformerDecoder` has
unrestricted cross-attention and is therefore accepted for smoothing only.

**Filtering here is conditional filtering given invariant parameters.** The
global head still uses `x_1:T`; integrating its draws does not give strict
online `p(theta_t | x_1:t)`. For strict online filtering, call the entire
approximator on each observation prefix so the invariant head also uses only
that prefix. Likewise, joint filtering factors define a normalized sequential
product, rather than a joint smoothing posterior conditioned on all data.
