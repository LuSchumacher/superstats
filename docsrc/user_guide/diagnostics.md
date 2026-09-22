# Diagnostics

Diagnostics follow the two stages of a Superstats analysis. Before inference, the question is whether the generative assumptions are sensible. These checks are methods of `Model`, because they require only the prior, transitions, links, and simulator. After inference, the question is whether the learned posterior is accurate, calibrated, informative, and capable of reproducing the data. These checks are methods of `Workflow`, which knows the model, posterior approximator, and division between time-varying and time-invariant parameters.

This separation prevents a common mistake: a plausible posterior cannot rescue an implausible generative model, and a plausible prior predictive distribution does not establish that the posterior approximation is reliable. Both stages need to be checked.

## Diagnostic overview

| Stage | Question | Method |
|---|---|---|
| Before inference | Are stochastic trajectories plausible on the inference scale? | `model.plot_time_varying_prior()` |
| Before inference | Are shared parameters and transition hyperparameters plausible? | `model.plot_time_invariant_prior()` |
| Before inference | Are all inferred quantities jointly plausible? | `model.plot_joint_prior()` |
| Before inference | Does the complete model produce plausible observations? | `model.plot_push_forward()` |
| After inference | Did optimization converge without a validation gap? | `workflow.plot_history()` |
| After inference | Are time-varying parameters recovered across the sequence? | `workflow.verify_time_varying()` |
| After inference | What happens at selected steps? | `workflow.recovery_at_steps()`, `calibration_at_steps()`, `z_score_contraction_at_steps()` |
| After inference | Are time-invariant parameters recovered and calibrated? | `workflow.verify_time_invariant()` |
| After inference | What do fitted posteriors look like? | `workflow.plot_time_varying_posterior()`, `plot_marginals()`, `plot_pairs()`, `plot_forest()` |
| After inference | Can posterior draws reproduce the observations? | `workflow.resimulate()` and `sup.diagnostics.plot_posterior_resimulation()` |

No single plot validates a model. Prior checks assess the specification, simulation-based checks assess amortized inference over the prior, posterior plots describe particular fitted datasets, and posterior predictive checks assess implications on the observation scale.

### A practical order

1. Check raw trajectory and coefficient priors with the prior plotting methods.
2. Check observable implications with `model.plot_push_forward()`.
3. Train the approximator and inspect `workflow.plot_history()`.
4. Run recovery and calibration on fresh prior-predictive simulations.
5. Inspect posteriors for representative and difficult datasets.
6. Run posterior predictive checks on every important observed variable.
7. Revise the model, prior, links, or training setup when a check fails, then repeat the sequence.

## Example model

The runnable example uses a DDM with a stochastic drift trajectory and invariant boundary separation and non-decision time. Transitions produce raw trajectories; the model links the final parameters into valid simulator ranges. The small simulation and training budgets keep the notebook practical, but they should be increased for a substantive analysis.

```python
import numpy as np
import superstats as sup

SEED = 2026
NUM_STEPS = 100
np.random.seed(SEED)
```

```python
prior = sup.JointPrior(
    v=sup.transition.RandomWalk(
        initial_prior=sup.Prior("normal", loc=0.0, scale=0.5),
        sigma=sup.Prior("halfnormal", scale=0.15),
        delta=0.0,
    ),
    a=sup.Prior("normal", loc=0.0, scale=0.5),
    tau=sup.Prior("normal", loc=-1.0, scale=0.4),
    bias=0.5,
)

model = sup.Model(
    prior=prior,
    simulator=sup.simulation.sample_ddm,
    latent_link_functions={
        "v": sup.LinkFunction("scaled_sigmoid", bounds=(-3.0, 3.0)),
        "a": sup.LinkFunction("scaled_sigmoid", bounds=(0.6, 2.2)),
        "tau": sup.LinkFunction("scaled_sigmoid", bounds=(0.1, 0.6)),
    },
    missing=None,
    contamination=None,
)

print("Time-varying targets:", model.local_keys)
print("Time-invariant targets:", model.hyper_keys + model.shared_keys)
```

## Before inference: diagnose the model

Run model diagnostics before spending time on training. The three prior plots show inference targets on their raw inference scale. They do not resolve formulas or context, apply `LinkFunction` objects, or call the observation simulator. The push-forward plot then checks the complete model after those operations have been applied.

### Time-varying prior trajectories

`plot_time_varying_prior()` displays stochastic local trajectories and optional marginal distributions. Inspect their starting values, variability, smoothness, drift, jumps, and behavior near the beginning and end of the sequence. Because it shows the raw inference scale, bounded simulator values are checked later with `plot_push_forward()`.

```python
fig = model.plot_time_varying_prior(
    num_steps=NUM_STEPS,
    num_trajectories=20,
)
```

### Time-invariant priors

`plot_time_invariant_prior()` displays shared parameters and inferred transition hyperparameters, such as the random-walk innovation scale.

```python
fig = model.plot_time_invariant_prior(
    num_draws=1_000,
    num_steps=NUM_STEPS,
)
```

### Joint inference prior

`plot_joint_prior()` combines both views. It also constructs deterministic trajectories from sampled curve coefficients, making it the most complete check of the quantities learned by the posterior approximator. Formula-derived parameters, fixed parameters, simulator defaults, and context do not appear because they are not separate inference targets.

```python
fig = model.plot_joint_prior(
    num_steps=NUM_STEPS,
    num_trajectories=20,
    num_draws=1_000,
)
```

### Prior push-forward checks

`plot_push_forward()` samples the complete generative model. It resolves transitions, formulas, and context, applies model-level links, calls the simulator, and applies configured contamination. Use it to assess observable implications such as response-time ranges, category frequencies, temporal trends, and between-dataset variation. Missingness is omitted by default; pass `apply_missing=True` when the missing-data pattern is itself part of the check.

```python
fig = model.plot_push_forward(
    batch_size=8,
    num_steps=NUM_STEPS,
    data_dim="response_time",
    kind="dist",
    num_cols=4,
)
```

```python
fig = model.plot_push_forward(
    batch_size=100,
    num_steps=NUM_STEPS,
    data_dim="choice",
    kind="time_series",
    aggregation=np.mean,
    uncertainty_fun="ci",
)
```

## During training: inspect optimization

The workflow owns all post-model diagnostics. This example uses marginal smoothing with separate posterior heads for the drift trajectory and invariant parameters.

```python
workflow = sup.Workflow(
    model=model,
    approximator="marginal",
    mode="smoothing",
)

train_data = model.sample(batch_size=20_000, num_steps=NUM_STEPS)
validation_data = model.sample(batch_size=200, num_steps=NUM_STEPS)
```

```python
history = workflow.fit_offline(
    data=train_data,
    validation_data=validation_data,
    epochs=50,
    batch_size=32,
)
```

`plot_history()` shows the training and validation losses. Look for stable optimization and a validation curve that does not progressively separate from training. Loss curves do not establish recovery or calibration.

```python
fig = workflow.plot_history(history)
```

## After inference: validate on held-out simulations

Recovery and calibration require known ground truth. Generate fresh simulations that were not used for training, then infer their parameters. These checks evaluate performance over the prior predictive distribution and should be repeated for scientifically important subsets when necessary.

```python
targets = model.sample(batch_size=200, num_steps=NUM_STEPS)
estimates = workflow.sample(
    data=targets,
    num_samples=300,
    batch_size=5,
)
```

### Time-varying verification

`verify_time_varying()` plots four quantities at every step: correlation across simulations, normalized RMSE relative to a prior-only baseline, posterior contraction relative to prior variance, and calibration error from empirical credible-interval coverage. Strong contraction is useful only when recovery and calibration are also adequate.

```python
fig = workflow.verify_time_varying(
    targets=targets,
    estimates=estimates,
)
```

### Diagnostics at selected steps

The selected-step methods accept zero-based indices. `recovery_at_steps()` compares true values with posterior point estimates, `calibration_at_steps()` compares nominal and empirical coverage, and `z_score_contraction_at_steps()` combines bias-sensitive posterior z-scores with informativeness.

```python
selected_steps = [0, NUM_STEPS // 2, NUM_STEPS - 1]

fig_recovery = workflow.recovery_at_steps(
    targets, estimates, time_steps=selected_steps
)
fig_calibration = workflow.calibration_at_steps(
    targets, estimates, time_steps=selected_steps
)
fig_contraction = workflow.z_score_contraction_at_steps(
    targets, estimates, time_steps=selected_steps
)
```

### Time-invariant verification

`verify_time_invariant()` produces recovery, calibration, and z-score-versus-contraction figures for shared parameters and inferred transition hyperparameters. Mixture parameters are expanded into named components. Pass `uncertainty_agg=None` to omit recovery intervals.

```python
fig_recovery, fig_calibration, fig_contraction = (
    workflow.verify_time_invariant(
        targets=targets,
        estimates=estimates,
    )
)
```

### Numerical diagnostics

The workflow methods are the recommended plotting interface. The four underlying time-varying metrics are also public when their values are needed for tables or comparisons. Estimates have shape `(simulations, posterior_samples, steps, parameters)` and targets have shape `(simulations, steps, parameters)`.

```python
v_estimates = estimates["v"]
v_targets = targets["v"]

metric_values = {
    "correlation": sup.diagnostics.correlation_per_step(v_estimates, v_targets),
    "nrmse": sup.diagnostics.nrmse_per_step(v_estimates, v_targets),
    "contraction": sup.diagnostics.posterior_contraction_per_step(v_estimates, v_targets),
    "calibration_error": sup.diagnostics.calibration_error_per_step(v_estimates, v_targets),
}
{name: values.shape for name, values in metric_values.items()}
```

## After inference: inspect fitted posteriors

Ground truth is normally unavailable for empirical data. Posterior plots then describe uncertainty and dependence for particular datasets, but they should be interpreted together with held-out simulation diagnostics. Here the simulated targets remain visible so the plots can be checked directly.

### Time-varying posteriors

`plot_time_varying_posterior()` shows posterior centers and uncertainty bands over time. Select individual datasets with `data_idx`, or aggregate across datasets with a callable such as `np.median`. Smoothing is a display operation and does not change the posterior model.

```python
fig = workflow.plot_time_varying_posterior(
    estimates=estimates,
    targets=targets,
    data_idx=0,
    uncertainty_fun="hdi",
    marginal=True,
)
```

### Time-invariant posterior marginals

`plot_marginals()` displays one-dimensional posteriors for one dataset. It uses the model's shared and hyperparameter keys and understands named mixture components. When several datasets are present, `data_idx` is required.

```python
fig = workflow.plot_marginals(
    estimates=estimates,
    targets=targets,
    data_idx=0,
    dist_type="both",
)
```

### Pair plots

`plot_pairs()` shows marginal distributions and pairwise dependence for the time-invariant posterior of one dataset. It can expose tradeoffs and non-identifiability hidden by univariate marginals.

```python
fig = workflow.plot_pairs(
    estimates=estimates,
    targets=targets,
    data_idx=0,
)
```

### Forest plots

`plot_forest()` compares time-invariant posteriors across datasets. Without aggregation, every selected dataset is retained as a row. An aggregation callable creates a visual population summary; it does not fit a hierarchical model.

```python
fig = workflow.plot_forest(
    estimates=estimates,
    targets=targets,
    data_idx=[0, 1, 2, 3, 4],
)
```

## Posterior predictive checks

Posterior predictive checks return to the observation scale. `Workflow.resimulate()` preserves the relationship between paired posterior parameters and the model, reconstructs deterministic trajectories, resolves formulas and links, and calls the simulator. For regression models, pass the original design through `context`. Missingness is omitted by default and can be enabled with `apply_missing=True`.

```python
selected_data = [0, 1, 2]
prediction = workflow.resimulate(
    estimates,
    num_sims=100,
    rng=SEED,
    data_idx=selected_data,
)
empirical = {key: value[selected_data] for key, value in targets.items()}
```

The final plot is stateless because it only compares two prepared data mappings. Repeat this comparison for every observed variable and relevant summary; matching one marginal distribution is not sufficient.

```python
fig = sup.diagnostics.plot_posterior_resimulation(
    prediction=prediction,
    empirical=empirical,
    data_dim="response_time",
    kind="dist",
)
```

```python
fig = sup.diagnostics.plot_posterior_resimulation(
    prediction=prediction,
    empirical=empirical,
    data_dim="choice",
    kind="time_series",
    aggregation=np.mean,
    smoothing="sma",
    smoothing_window=5,
)
```
