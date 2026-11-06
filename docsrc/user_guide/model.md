# Model

The `Model` class connects a simulator to its `JointPrior`. It draws parameter values, broadcasts time-invariant values across steps, calls the simulator in batches, and returns observations and parameters in one named dictionary. It can also apply optional contamination and missingness processes.

## Example

We continue with the DDM example. The prior lets drift rate, boundary separation, and non-decision time vary independently across observations, while starting bias stays fixed.

```python
import superstats as sup
```

```python
simulator = sup.simulation.sample_ddm
```

```python
prior = sup.JointPrior(
    v=sup.transition.RandomWalk(),
    a=sup.transition.RandomWalk(),
    tau=sup.transition.RandomWalk(),
    bias=0.5,
)
```

```python
model = sup.Model(
    simulator=simulator,
    prior=prior,
    latent_link_function={
        "v": sup.LinkFunction(bounds=(0.0, 6.0)),
        "a": sup.LinkFunction(bounds=(0.2, 4.0)),
        "tau": sup.LinkFunction(bounds=(0.0, 2.0)),
    },
    missing=None,
    contamination=None,
)
```

This example disables both augmentation processes. This is explicit because `Model` otherwise defaults to `missing="random"`. We now draw 100 independent parameter configurations and simulate a 200-observation sequence from each.

## Link Functions

Transitions produce trajectories on an unconstrained scale. `Model` provides two explicit places to transform them:

- `latent_link_function` transforms sampled prior parameters before formulas are resolved. Its keys name parameters in `JointPrior`, such as a time-varying regression coefficient `dv_t`.
- `formula_link_function` transforms final parameters produced by `Formula` after the complete formula has been resolved. Its keys must name formula targets that are passed to the simulator.

Parameters omitted from either mapping are passed through unchanged. Both stages transform copies used by the simulation pipeline: `Model.sample()` continues to return the original unconstrained prior draws as inference targets.

Formulas are evaluated in their declared order. All latent links are applied
before the first formula and all formula links are applied only after the last
formula. Consequently, a later formula sees the unlinked output of an earlier
formula target. Simulator defaults and context-bound simulator parameters are
assumed to already be on the simulator scale; `Model` does not infer a domain
from a parameter name.

For example, with `formula=Formula(["v = v_0 + dv_t * validity"])`, configuring a latent link for `dv_t` gives `v = h(dv_t) * validity + v_0`, whereas configuring a formula link for `v` gives `v = h(v_0 + dv_t * validity)`. Both mappings may be supplied when both transformations are part of the model. Parameters that come directly from `JointPrior`, including parameters used without a formula, always use latent links.

`LinkFunction` supports the following named transformations:

| Link | Domain | Typical use |
|---|---|---|
| `LinkFunction("scaled_sigmoid", bounds=(lower, upper))` | Finite interval | Smoothly bounded stochastic trajectories |
| `LinkFunction("clip", bounds=(lower, upper))` | Finite interval | Deterministic trajectories that should remain unchanged inside the interval |
| `LinkFunction("softplus")` | Positive values | Positive parameters without a finite upper bound |
| `LinkFunction("exp")` | Positive values | Exponential-scale parameters |
| `LinkFunction("identity")` | Unconstrained | Parameters that need no transformation |

As a practical default, use a scaled sigmoid with sensible bounds for stochastic trajectories and hard clipping with sensible bounds for deterministic trajectories.

Bounds must contain two finite, strictly increasing values. They are required
for `"clip"` and otherwise apply only to `"scaled_sigmoid"`. Custom callables
are also accepted; they must preserve the input shape and return finite values.
The scaled sigmoid is numerically stable for large predictors, although
floating-point rounding can still produce an interval endpoint. Use a strictly
positive lower bound when the simulator does not accept zero.

Links change the values used to construct simulator parameters, not what the posterior approximator learns. `Model.sample()` returns stochastic and reconstructed deterministic trajectories on their raw scale, together with the raw coefficients and transition hyperparameters used as inference targets, while formulas and the simulator receive the configured linked copies. This keeps inference unconstrained and makes posterior resimulation apply the same links consistently. Simulator defaults and context-bound simulator parameters are expected to already be on the simulator scale. `Model.sample_prior()["model_params"]` can be used to inspect the final simulator parameters directly. Prior plots remain on the raw scale; prior push-forward plots show the consequences of both link stages in simulated observations.

## Synthetic data generation

```python
sim_data = model.sample(
    batch_size=100,
    num_steps=200,
)

print(sim_data.keys())
```

Every observation and parameter has its own dictionary key. Here, `response_time`, `choice`, and `time_steps` have shape `(100, 200)`. The local parameters `v`, `a`, and `tau` have shape `(100, 200, 1)` on their raw inference scale, while their transition scales have shape `(100, 1)`. The DDM receives the linked versions of `v`, `a`, and `tau`. The fixed `bias` is used by the simulator but is not returned as an inference target.

## Prior Push-forward Checks

Before training, check whether the combined prior and simulator generate credible observations. A prior push-forward check can reveal impossible values, excessive variability, unrealistic tails, or temporal patterns that do not match domain knowledge. Revise the model assumptions when it fails; more neural-network training cannot repair an implausible generative model.

Simulate five datasets (`batch_size=5`) each with 200 observations (`num_steps=200`) and plot per dataset (`aggregation=None`) response time (`data_dim=0`) distributions (`kind="dist"`).

```python
fig = model.plot_push_forward(
    batch_size=5,
    num_steps=200,
    data_dim="response_time",
    kind="dist",
    dist_type="hist",
    aggregation=None,
)
```

Alternatively, we can plot choices (`data_dim=1`) as time series (`kind="time_series"`) aggregated across datasets by median (`aggregation=np.median`). Additionally, we depict variation as standard deviation (`uncertainty_fun="std"`), show individual datasets (`spaggetti=True`), and display the marginal distribution (`marginal=True`) as a kernel density (`dist_type="kde"`).

Disclaimer: we could choose better settings for plotting push forward checks for choices. Here, I simply wanted to demonstrate the flexible possibilities of the `plot_push_forward()` method.

```python
import numpy as np
```

```python
fig = model.plot_push_forward(
    batch_size=5,
    num_steps=200,
    data_dim="choice",
    kind="time_series",
    aggregation=np.mean,
    uncertainty_fun="ci",
    spaghetti=True,
    marginal=True,
    dist_type="kde",
)
```
