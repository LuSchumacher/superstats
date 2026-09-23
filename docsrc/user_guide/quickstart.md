# Quickstart

This notebook builds a small diffusion decision model, checks its simulations, and shows where training and empirical data enter the workflow. Run the cells in order.

## 1. Choose the sequence length

Install Superstats with `pip install superstats`, then import it and choose the number of ordered observations per dataset. Use the empirical sequence length you intend to analyze. The default recurrent workflow expects inference data to have the same length used for training.

```python
import superstats as sup

NUM_STEPS = 100
```

## 2. Define how parameters behave

The built-in DDM simulator expects `v`, `a`, `tau`, and `bias`. Here, drift rate varies according to a random walk bounded by a model-level link, while the other parameters are time-invariant.

```python
prior = sup.JointPrior(
    v=sup.transition.RandomWalk(
        sigma=sup.Prior("halfnormal", scale=0.15),
        delta=0.0,
    ),
    a=sup.Prior("halfnormal", scale=1.0, shift=0.5),
    tau=sup.Prior("halfnormal", scale=0.3),
    bias=0.5,
)
```

This declaration asks the posterior approximator to estimate the trajectory `v`, its volatility `v_sigma`, and one shared value each for `a` and `tau`. The fixed `bias` is passed to the simulator but is not estimated.

## 3. Build and inspect the generative model

```python
model = sup.Model(
    prior=prior,
    simulator=sup.simulation.sample_ddm,
    latent_link_function={"v": sup.LinkFunction(bounds=(-4.0, 4.0))},
    missing=None,
    contamination=None,
)

simulated = model.sample(batch_size=8, num_steps=NUM_STEPS)

print(simulated["response_time"].shape)  # (8, 100)
print(simulated["choice"].shape)         # (8, 100)
print(simulated["v"].shape)              # (8, 100, 1)
print(simulated["v_sigma"].shape)        # (8, 1)
```

`Model` defaults to random missingness. This example sets `missing=None` explicitly so the simulations contain only DDM observations. See [Data augmentation](augmentation.md) before enabling missingness or contamination.

Plot both the prior trajectories and the observations they imply.

```python
model.plot_joint_prior(num_steps=NUM_STEPS, num_trajectories=20)

model.plot_push_forward(
    batch_size=20,
    num_steps=NUM_STEPS,
    data_dim="response_time",
    kind="dist",
)
```

Look for plausible parameter ranges, response times, choice proportions, between-dataset variation, and temporal behavior. Revise the priors before training if the simulated world does not resemble the one you intend to study.

## 4. Create and train a workflow

```python
workflow = sup.Workflow(
    model=model,
    checkpoint_filepath="checkpoints/ddm",
)

history = workflow.fit_online(
    num_steps=NUM_STEPS,
    epochs=2,
    num_batches_per_epoch=10,
    batch_size=16,
)
```

The small budget above is a smoke test for the pipeline, not enough for a scientific analysis. Increase the simulation and optimization budget until loss curves stabilize and held-out recovery and calibration are satisfactory. The checkpoint directory lets later sessions restore the fitted approximator.

## 5. Verify before fitting real data

Verification must use fresh simulations that were not reused from offline training.

```python
targets = model.sample(batch_size=100, num_steps=NUM_STEPS)
estimates = workflow.sample(targets, num_samples=500, batch_size=4)

workflow.verify_time_varying(targets=targets, estimates=estimates)
workflow.verify_time_invariant(targets=targets, estimates=estimates)
```

Do not treat empirical estimates as trustworthy until recovery, contraction, and calibration are adequate for the parameters and time points you plan to interpret.

## 6. Format and fit empirical data

Inference data use the simulator's output names and keep a dataset axis, even for one dataset. To keep this notebook runnable, the next cell treats the first simulated dataset as a stand-in for empirical data. Replace these arrays with your observations.

```python
observed = {
    "response_time": simulated["response_time"][:1],
    "choice": simulated["choice"][:1],
}

posterior = workflow.sample(observed, num_samples=1_000, batch_size=1)
```

Both arrays above have shape `(1, NUM_STEPS)`. For multiple participants, stack one row per participant. If your data are in a long-format pandas `DataFrame`, `workflow.prepare_data(...)` can group rows and preserve time order. To pad unequal sequences or represent missing observations, first train a model with a compatible missingness process; the missingness mask must be part of the workflow's training conditions.

Finally, re-simulate observations from the posterior and compare them with the data before interpreting trajectories.

```python
prediction = workflow.resimulate(posterior, num_sims=100)
sup.diagnostics.plot_posterior_resimulation(
    prediction=prediction,
    empirical=observed,
)
```

Continue with [Core concepts](introduction.md) for the reasoning behind this workflow, or open the [minimal workflow demo](https://github.com/LuSchumacher/superstats/blob/main/examples/minimal_workflow_demo.ipynb) for a more complete analysis.
