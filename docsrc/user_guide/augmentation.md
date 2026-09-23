# Data Augmentation

Real data often contain missing observations, lapses, guesses, or unusually fast and slow responses. If these features are expected in the empirical data, they should also occur in the simulations used to train the neural approximator. Otherwise, the approximator is asked to fit data that differ systematically from its training distribution.

Superstats supports two data augmentation processes in the `Model` class:

- **Missingness** removes a complete observation at a time step and records it in a `missing_mask`.
- **Contamination** replaces selected observations with draws from a contaminant distribution.

Both processes are optional modeling assumptions. Their probabilities should describe plausible imperfections in the application and should be checked through prior push-forward simulations. Contamination is applied first and missingness second.

## Example

Let us continue with the DDM example from the previous pages. We first define the observation model and joint prior without attaching an augmentation process.

```python
import numpy as np

import superstats as sup
```

```python
simulator = sup.simulation.sample_ddm

prior = sup.JointPrior(
    v=sup.transition.RandomWalk(),
    a=sup.Prior("halfnormal", scale=1.0, shift=0.5),
    tau=sup.Prior("halfnormal", scale=0.3),
    bias=0.5,
    p_missing=sup.Prior("beta", a=1.5, b=15.0),
)
```

## Missing Observations

`RandomMissingProcess` implements missing completely at random (MCAR). For every simulated dataset, it draws a missingness probability and independently decides whether each time step is missing. When a time step is selected, every observed variable at that step is replaced by `missing_value`. For the DDM, response time and choice are therefore always missing together.

By default, `Model` uses `missing="random"`, which constructs `RandomMissingProcess()` with the package's default prior for `p_missing`. Set `missing=None` to disable missingness. Here, we specify the process explicitly and use a Beta prior so that the missing proportion can differ across simulated datasets.

```python
missing_process = sup.simulation.RandomMissingProcess(
    missing_value=-1,
)

model_missing = sup.Model(
    simulator=simulator,
    prior=prior,
    latent_link_function={
        "v": sup.LinkFunction(bounds=(-4.0, 4.0)),
    },
    missing=missing_process,
    contamination=None,
)
```

```python
missing_data = model_missing.sample(
    batch_size=4,
    num_steps=100,
    rng=np.random.default_rng(42),
)

print("Missing mask shape:", missing_data["missing_mask"].shape)
print("Missing observations per dataset:", missing_data["missing_mask"].sum(axis=1))
print("Missing probabilities:", missing_data["p_missing"].squeeze())
```

The output contains the augmented observation variables, a Boolean `missing_mask` with shape `(batch_size, num_steps)`, and the sampled `p_missing` for each dataset. The mask is included in the summary variables used by the default workflow adapter, allowing the neural network to distinguish a sentinel value from an actual observation. `p_missing` describes the augmentation process but is never an inference target; its transition hyperparameters are excluded as well. The beta prior already produces valid probabilities, so it does not need a link.

### Controlling the Missingness Process

The missingness probability can be specified in two ways:

- A scalar such as `p_missing=0.1` fixes the probability for all simulated datasets.
- A `Prior`, such as `Prior("beta", a=1.5, b=15)`, draws a different probability for every dataset.

Set `shared_across_batch=True` to draw one probability and one mask that are shared by the entire simulated batch. This is mainly useful when the batch represents repeated versions of the same design. The default, `False`, generates independent missingness patterns for independent datasets.

Missingness is applied by default in `Model.sample()`, including simulations
used for training. Prior push-forward plots and posterior resimulation produce
complete observations by default; pass `apply_missing=True` to include the
configured missingness process. Posterior resimulation then draws a fresh
missingness probability from its prior because `p_missing` is never estimated.
Direct calls to a missingness process must receive `probability` explicitly.

`missing_value` may be one scalar for every observation variable, a mapping from observation names to values, or an array containing one value per observation variable. Choose sentinels that cannot be confused with valid observations.

## Contaminated Responses

`RandomChoiceContamination` implements the random-choice contaminant process described by [Wu, Radev, and Tuerlinckx (2026)](https://arxiv.org/abs/2412.20586). At every selected time step, it replaces both DDM outputs:

- Response times are drawn from a heavy-tailed Student's $t$ distribution on the log-response-time scale. Its location and scale are adapted to each simulated dataset.
- Discrete choices are sampled from the choice values observed in the batch. Continuous choices are sampled uniformly over their observed range.

Non-positive response times, such as DDM timeouts, are left unchanged. Contamination is disabled by default; use `contamination="random_choice"` for the package defaults or pass a configured process as shown below.

```python
contamination_prior = sup.JointPrior(
    **prior.params,
    p_contaminated=sup.Prior("beta", a=1.5, b=15.0),
)
contamination_process = sup.simulation.RandomChoiceContamination(infer=False)

model_contaminated = sup.Model(
    simulator=simulator,
    prior=contamination_prior,
    latent_link_function={
        "v": sup.LinkFunction(bounds=(-4.0, 4.0)),
    },
    missing=None,
    contamination=contamination_process,
)

contaminated_data = model_contaminated.sample(batch_size=4, num_steps=100)
print("Contamination probabilities:", contaminated_data["p_contaminated"])
```

With `infer=False`, `p_contaminated` is a nuisance variable: it varies across training simulations and is returned as augmentation metadata, but the neural approximator is not trained to estimate it. This is appropriate when contamination should make inference robust without being scientifically interpreted.

### Inferring the Contamination Probability

The contamination probability can be fixed, drawn once per dataset, or allowed to change over time. Set `infer=True` to register it with the `Model`. Its specification determines the parameter category, following the same rules as `JointPrior`:

| Specification | Parameter category | Interpretation |
|---|---|---|
| Scalar | `fixed_param` | Known probability; not estimated because it has no uncertainty |
| `Prior` | `shared_param` | One estimated probability per dataset |
| `StochasticTransition` | `local_param` | Estimated probability trajectory, plus estimated transition hyperparameters |
| `DeterministicTransition` | `deterministic_param` | Structured probability trajectory; its prior-valued transition parameters are estimated |

A probability transition must be paired with a `Model` link whose bounds fall within `[0, 1]`. The following example lets contamination change gradually across time according to a random walk.

```python
time_varying_contamination_prior = sup.JointPrior(
    **prior.params,
    p_contaminated=sup.transition.RandomWalk(
        initial_prior=sup.Prior("normal", loc=-2.0, scale=0.5),
        sigma=sup.Prior("halfnormal", scale=0.05),
        delta=0.0,
    ),
)
time_varying_contamination = sup.simulation.RandomChoiceContamination(infer=True)

model_time_varying_contamination = sup.Model(
    simulator=simulator,
    prior=time_varying_contamination_prior,
    latent_link_function={
        "v": sup.LinkFunction(bounds=(-4.0, 4.0)),
        "p_contaminated": sup.LinkFunction(bounds=(0.0, 0.25)),
    },
    missing=None,
    contamination=time_varying_contamination,
)
```

```python
dynamic_data = model_time_varying_contamination.sample(
    batch_size=4,
    num_steps=100,
    tile_to_steps=True,
)

print("Local parameters:", model_time_varying_contamination.local_keys)
print("Hyperparameters:", model_time_varying_contamination.hyper_keys)
print("p_contaminated shape:", dynamic_data["p_contaminated"].shape)
```

Here, `p_contaminated` is a local parameter with shape `(batch_size, num_steps, 1)`, while `p_contaminated_sigma` is a time-invariant hyperparameter. The default workflow adapter automatically includes both as inference variables. Fixed transition parameters such as `p_contaminated_delta` remain excluded from inference.

During posterior resimulation, an inferred contamination probability is taken
from the paired posterior draw. When `infer=False`, contamination remains
active but its probability is drawn afresh from the prior. Calling the
contamination process directly, outside `Model`, requires probabilities that
are already within `[0, 1]`.

## Combining Missingness and Contamination

Missingness and contamination can be used together. Superstats first contaminates the simulated observations and then applies missingness. Consequently, an observation selected by both processes is represented as missing in the final data. The returned dictionary retains `p_contaminated`, `missing_mask`, and `p_missing`.

```python
model_augmented = sup.Model(
    simulator=simulator,
    prior=contamination_prior,
    latent_link_function={
        "v": sup.LinkFunction(bounds=(-4.0, 4.0)),
    },
    missing=missing_process,
    contamination=contamination_process,
)

augmented_data = model_augmented.sample(
    batch_size=4,
    num_steps=100,
    rng=np.random.default_rng(42),
)

print(
    {
        key: augmented_data[key].shape
        for key in ("response_time", "choice", "missing_mask", "p_missing", "p_contaminated")
    }
)
```

Before training, use `model.plot_push_forward()` to compare the augmented simulations with realistic data characteristics: missing proportions, response-time tails, choice frequencies, and changes over time. Augmentation cannot repair a misspecified observation model; it should represent specific imperfections that are plausible in the data collection process.

## Custom Augmentation Processes

Custom processes can be passed as callable objects. A missingness callable receives the named data dictionary and must return the modified observations together with a Boolean `missing_mask`. A contamination callable returns the modified observations and may include additional metadata. Both may accept an `rng` argument.

```python
def custom_missing(data: dict, rng=None) -> dict:
    ...
    return {**filled_data, "missing_mask": mask}

def custom_contamination(data: dict, rng=None) -> dict:
    ...
    return {**contaminated_data, "custom_metadata": values}
```

For reusable implementations, subclass `MissingProcess` or `ContaminationProcess` and implement `apply()`.
