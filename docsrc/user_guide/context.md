# Context

Context represents variables that are defined outside the model parameters but
vary across trials, datasets, or experimental conditions. Examples include
stimulus difficulty, reward, condition labels, available options, and the index
of a correct response.

Superstats separates two questions:

1. **Where does context come from?** It can be fixed or generated anew for
   every simulated dataset.
2. **Where is context used?** Design context enters parameter formulas, while
   simulator context is passed to the observation simulator.

The same generated variable may be used by both consumers. All generated
context is returned by `Model.sample()` and included in the workflow's summary
variables, even when a variable is not selected for either consumer.

## Design context and simulator context

| Context role | Selected with | Destination | Typical examples |
|---|---|---|---|
| Design context | `design_context=(...)` | `Formula` | difficulty, condition, reward, trial type |
| Simulator context | `simulator_context=(...)` | Simulator argument or its `context` keyword | stimulus arrays, available options, correct-response index |

A simulator-context name matching a named simulator argument is bound directly
to that argument. Other selected names are forwarded in the simulator's
`context` mapping, so the simulator must accept a `context` keyword.
Context-bound parameters replace prior or formula values with the same
simulator-parameter name. They are expected to already be on the simulator
scale.

The processing order is therefore:

1. Sample model parameters and context.
2. Apply latent links to configured prior parameters.
3. Evaluate formulas with design context.
4. Apply formula links to configured formula targets.
5. Bind matching simulator-context variables and collect the remaining
   simulator context.
6. Call the simulator.

The examples below use the following imports:

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import superstats as sup

SEED = 2026
rng = np.random.default_rng(SEED)
```

## A minimal routing example

This small simulator makes both routes visible. `difficulty` changes the
parameter `theta` through a formula. `condition` matches a named simulator
argument and is bound directly. `offset` does not match a simulator argument
and is therefore passed through the simulator's `context` mapping.

```python
def observation_model(theta, condition, *, context):
    observation = theta + condition + context["offset"].reshape(-1)
    return {"observation": observation}
```

## Fixed context

A mapping defines one fixed trial sequence. Each value may be a scalar, which
is repeated over all trials, or an array whose first dimension equals
`num_steps`. The complete sequence is repeated across simulated datasets. This
is useful when every simulated participant should receive the same experimental
design.

```python
fixed_context = {
    "difficulty": [-1.0, 0.0, 1.0, 2.0],
    "condition": [0.0, 1.0, 0.0, 1.0],
    "offset": 0.1,
    "unused": [5.0, 5.0, 5.0, 5.0],
}

fixed_model = sup.Model(
    prior=sup.JointPrior(intercept=1.0, slope=0.5),
    simulator=observation_model,
    formula=sup.Formula([
        "theta = intercept + slope * difficulty",
    ]),
    context_simulator=fixed_context,
    design_context=("difficulty",),
    simulator_context=("condition", "offset"),
    missing=None,
    contamination=None,
)

fixed_sample = fixed_model.sample(batch_size=2, num_steps=4)

print(fixed_sample["observation"])
print("Context keys:", fixed_model.context_keys)
print("Summary keys:", fixed_model.summary_keys)
```

The unselected variable `unused` is neither part of the formula nor passed to
the simulator. It is nevertheless returned and included in `summary_keys`.
Selectors control generative routing; they do not remove variables from the
data seen by the posterior encoder. Remove unwanted variables from the context
source itself.

A pandas `DataFrame` is another representation of the same fixed design.
Column names become context names and rows become trials.

```python
context_frame = pd.DataFrame({
    "difficulty": [-1.0, 0.0, 1.0, 2.0],
    "condition": [0.0, 1.0, 0.0, 1.0],
    "offset": [0.1, 0.1, 0.1, 0.1],
})

frame_model = sup.Model(
    prior=sup.JointPrior(intercept=1.0, slope=0.5),
    simulator=observation_model,
    formula=sup.Formula([
        "theta = intercept + slope * difficulty",
    ]),
    context_simulator=context_frame,
    design_context=("difficulty",),
    simulator_context=("condition", "offset"),
    missing=None,
    contamination=None,
)

frame_sample = frame_model.sample(
    batch_size=2,
    num_steps=len(context_frame),
)
```

## Context sources

`Model(context_simulator=...)` accepts four forms:

| Source | Behavior | Expected shape |
|---|---|---|
| Mapping | One fixed design, repeated across datasets | scalar or `(num_steps, ...)` |
| `pandas.DataFrame` | One fixed design, with columns used as names | `(num_steps, num_columns)` |
| Batched callable | Generates fresh context for a complete batch | returns `(batch_size, num_steps, ...)` values |
| `ContextSimulator` | Explicit wrapper for batched or per-dataset generation | depends on `is_batched` |

A callable passed through `context_simulator` is automatically wrapped as a batched
`ContextSimulator`. It must accept `batch_size` and `num_steps` as keyword
arguments and return a mapping with consistent string keys.

## Simulating context

Use generated context when the experimental design or environment should vary
across simulated datasets. The following generator randomizes difficulty and
condition independently while keeping a dataset-specific offset constant across
its trials.

```python
def generate_context(*, batch_size, num_steps):
    dataset_offset = rng.normal(0.0, 0.1, size=(batch_size, 1))
    return {
        "difficulty": rng.choice(
            [-1.0, 0.0, 1.0],
            size=(batch_size, num_steps),
        ),
        "condition": rng.integers(
            0,
            2,
            size=(batch_size, num_steps),
        ),
        "offset": np.broadcast_to(
            dataset_offset,
            (batch_size, num_steps),
        ),
    }


generated_model = sup.Model(
    prior=sup.JointPrior(intercept=1.0, slope=0.5),
    simulator=observation_model,
    formula=sup.Formula([
        "theta = intercept + slope * difficulty",
    ]),
    context_simulator=generate_context,
    design_context=("difficulty",),
    simulator_context=("condition", "offset"),
    missing=None,
    contamination=None,
)

generated_sample = generated_model.sample(
    batch_size=3,
    num_steps=6,
)
```

### Explicit `ContextSimulator`

Use `ContextSimulator(..., is_batched=True)` when the function already
generates a full batch. This is also what a direct callable `context_simulator` uses
internally. Set `is_batched=False` when the function generates one dataset at a
time and accepts only `num_steps`; the wrapper calls it once per batch element
and stacks the results.

```python
def generate_one_context(*, num_steps):
    return {
        "difficulty": rng.normal(size=num_steps),
    }


unbatched_context = sup.ContextSimulator(
    generate_one_context,
    is_batched=False,
)
unbatched_draw = unbatched_context.sample(
    batch_size=3,
    num_steps=6,
)
```

A name may appear in both `design_context` and `simulator_context`. In that
case, the same draw is available to the formula and simulator. Context and
parameters share one namespace inside `Formula`, so a context name must not
shadow a coefficient or other parameter name. Selectors must be sequences such
as `("difficulty",)`, not a bare string. Missing selected names raise an error
when the model validates its pilot draw.

## Context as regressors

Regression uses `Formula` as the sole design API. A formula names a final
simulator parameter on the left and combines coefficients with design context
on the right. Formulas support numeric literals, parentheses, `+`, `-`, `*`,
`/`, and `**`. They are evaluated in order, so later formulas may use earlier
targets.

More generally, for design columns $X_{btk}$ and coefficients $B_{btkp}$,

$$
\eta_{btp} = \sum_k X_{btk}B_{btkp}, \qquad
\theta_{btp} = h_p(\eta_{btp}).
$$

A time-invariant coefficient is broadcast across trials, whereas a
time-varying coefficient retains its full trajectory. Intercepts and slopes
can vary independently; the output link does not depend on which coefficients
vary. With a formula link, coefficients are defined on the predictor scale. In
particular, at a zero-valued covariate the simulator parameter is $h_p(\beta_0)$,
not $\beta_0$, and coefficient priors should be chosen accordingly.

Interactions can be written directly, for example
`"a = a_0 + b_a * difficulty + b_interaction * difficulty * reward"`.
Categorical predictors must be supplied as named, pre-encoded dummy columns
with a fixed reference level. Superstats does not implicitly standardize
predictors or generate random design columns. Use the same coding and scaling
during simulation and inference.

The example below assigns one regressor to DDM drift and another to boundary
separation:

$$
v_t = h_v(\beta_{v,0} + \beta_{v,1}x_{v,t}), \qquad
a_t = h_a(\beta_{a,0} + \beta_{a,1}x_{a,t}).
$$

The coefficients are inferred on the raw predictor scale. `Model` applies the
links to the completed formula outputs immediately before calling the DDM.

```python
NUM_STEPS = 30


def generate_ddm_design(*, batch_size, num_steps):
    return {
        "x_drift": rng.choice(
            [-1.0, 1.0],
            size=(batch_size, num_steps),
        ),
        "x_boundary": rng.choice(
            [-1.0, 1.0],
            size=(batch_size, num_steps),
        ),
    }


ddm_formula = sup.Formula([
    "v = v_intercept + v_slope * x_drift",
    "a = a_intercept + a_slope * x_boundary",
])

ddm_prior = sup.JointPrior(
    v_intercept=sup.Prior("normal", loc=0.0, scale=0.6),
    v_slope=sup.Prior("normal", loc=0.8, scale=0.3),
    a_intercept=sup.Prior("normal", loc=0.0, scale=0.4),
    a_slope=sup.Prior("normal", loc=-0.6, scale=0.25),
    tau=0.25,
    bias=0.5,
)

ddm_model = sup.Model(
    prior=ddm_prior,
    simulator=sup.simulation.sample_ddm,
    formula=ddm_formula,
    context_simulator=generate_ddm_design,
    design_context=("x_drift", "x_boundary"),
    simulator_context=(),
    formula_link_functions={
        "v": sup.LinkFunction(
            "scaled_sigmoid",
            bounds=(-3.0, 3.0),
        ),
        "a": sup.LinkFunction(
            "scaled_sigmoid",
            bounds=(0.6, 2.2),
        ),
    },
    missing=None,
    contamination=None,
)

ddm_sample = ddm_model.sample(
    batch_size=4,
    num_steps=NUM_STEPS,
)

print("Time-varying inference targets:", ddm_model.local_keys)
print("Time-invariant inference targets:", ddm_model.shared_keys)
print("Formula-derived parameters:", ddm_model.formula_keys)
print("Workflow summaries:", ddm_model.summary_keys)
```

`v` and `a` vary across trials because their regressors vary, but they are
deterministic formula outputs rather than independent posterior targets. The
posterior learns the four free coefficients. Formula targets are returned at
their final linked values for inspection and are listed in
`model.formula_keys`; they are not added to the inference targets, summary
inputs, or raw-prior plots. Posterior resimulation reconstructs them from the
raw inferred coefficients and the supplied context.

```python
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].scatter(
    ddm_sample["x_drift"][0],
    ddm_sample["v"][0, :, 0],
    alpha=0.7,
)
axes[0].set(
    xlabel="x_drift",
    ylabel="linked drift v",
)
axes[1].scatter(
    ddm_sample["x_boundary"][0],
    ddm_sample["a"][0, :, 0],
    alpha=0.7,
)
axes[1].set(
    xlabel="x_boundary",
    ylabel="linked boundary a",
)
fig.tight_layout()
```

## Context during inference

The default adapter concatenates observations, time, and every generated
context variable into `summary_variables`. Consequently, the posterior is
conditioned on the design that generated each observation. The four
coefficients are time-invariant, so this model activates only the workflow's
invariant posterior head.

```python
workflow = sup.Workflow(
    model=ddm_model,
    approximator="marginal",
    mode="smoothing",
)

adapted = workflow.adapter(ddm_sample)
{key: value.shape for key, value in adapted.items()}
```

A small offline training run would then use the complete model output:

```python
train_data = ddm_model.sample(
    batch_size=1_000,
    num_steps=NUM_STEPS,
)
validation_data = ddm_model.sample(
    batch_size=200,
    num_steps=NUM_STEPS,
)

history = workflow.fit_offline(
    data=train_data,
    validation_data=validation_data,
    epochs=8,
    batch_size=64,
)
```

Increase the simulation budget and number of epochs for an application.

## Preserve context during posterior resimulation

When fitting observed data, posterior prediction should normally use the
original design rather than generate a new one. Pass the observed context to
`Workflow.resimulate()`. Batched context values have shape
`(datasets, steps, ...)`; the workflow selects `data_idx` and repeats the design
for the requested posterior simulations.

```python
observed = ddm_model.sample(
    batch_size=1,
    num_steps=NUM_STEPS,
)
posterior = workflow.sample(
    data=observed,
    num_samples=500,
    batch_size=1,
)

observed_context = {
    key: observed[key]
    for key in ("x_drift", "x_boundary")
}
prediction = workflow.resimulate(
    estimates=posterior,
    num_sims=100,
    rng=SEED,
    context=observed_context,
)
```

For empirical data, include every context variable as a named input alongside
the observations. `workflow.prepare_data()` currently reshapes observation
columns only; add context arrays from the DataFrame to the returned mapping with
the same dataset and trial ordering. The number and ordering of trials must
agree across observations and context.

Use the same coding and scaling at training and inference time. Changing a
regressor's reference level or scale changes the meaning of its coefficients.

## Common pitfalls

- A fixed mapping or DataFrame must contain exactly `num_steps` trial rows; it
  is then repeated across datasets.
- A batched generator must return arrays beginning with
  `(batch_size, num_steps)`. Use `is_batched=False` for one-dataset generators.
- `design_context` and `simulator_context` must be sequences, even when
  selecting one name: use `("x",)`.
- Every selected name must be produced by the context source.
- Non-matching simulator-context names require a simulator that accepts
  `context`.
- Context names used by `Formula` must not shadow parameter names.
- Formula coefficients live on the raw predictor scale unless they have a
  latent link. Formula links constrain final formula targets.
- Formula-derived parameters are returned for inspection but are not additional
  inference targets.
- Posterior resimulation should receive the original observed context when
  predictions must preserve the experimental design.
