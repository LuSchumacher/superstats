# Regression and link functions

`JointPrior` defines raw coefficients and their dynamics. `Model` resolves formulas, binds context supplied as simulator parameters, applies output links, and calls the simulator. The same resolution path is used for initialization, prior simulation, diagnostics, and simulation from posterior coefficients.

For design columns $X_{btk}$ and potentially time-varying coefficients $B_{btkp}$:

$$\eta_{btp}=\sum_k X_{btk}B_{btkp}, \qquad \theta_{btp}=h_p(\eta_{btp}).$$

A time-invariant coefficient broadcasts across trials. A time-varying coefficient keeps its full trajectory. Either the intercept, the slope, both, or neither may vary. The output link does not depend on this choice. `link_function` follows CogFormer's naming convention: it maps a linear predictor to a cognitive parameter (the inverse link in GLM terminology).

## A mixed regression model

```python
import superstats as sup

prior = sup.JointPrior(
    a_0=sup.transition.RandomWalk(
        initial_prior=sup.Prior("normal", loc=0.0, scale=0.3),
        sigma=0.05,
        delta=0.0,
    ),
    b_a=sup.Prior("normal", loc=0.0, scale=0.5),
    v=sup.Prior("normal"),
    tau=0.2,
    bias=0.5,
)

model = sup.Model(
    prior=prior,
    simulator=sup.simulation.sample_ddm,
    missing=None,
    context={"difficulty": [-1.0, 0.0, 1.0]},
    formula=sup.Formula(["a = a_0 + b_a * difficulty"]),
    design_context=("difficulty",),
    simulator_context=(),
    link_function={"a": sup.LinkFunction(bounds=(0.2, 4.0))},
)

simulated = model.sample(batch_size=8, num_steps=3)
```

`Formula` is the sole regression API. It expresses the design as readable arithmetic, including interactions such as `"a = a_0 + b_a * difficulty + b_interaction * difficulty * reward"`. Supply categorical predictors as named, pre-encoded dummy columns with a fixed reference level. No random design columns or implicit standardization are introduced.

`context` accepts a `ContextSimulator`, a batched callable accepting `batch_size` and `num_steps`, or fixed context supplied as a mapping or DataFrame. It produces named variables once per draw. `design_context` selects variables for Formula; `simulator_context` selects variables for the simulator. A variable may appear in both selectors. Unselected variables remain in model outputs, but are not sent to either consumer. Missing selected names raise an error.

Simulator context matching a simulator parameter, such as `correct_idx`, is bound to that parameter. Other selected variables are forwarded through the simulator's `context` keyword. Formula evaluation remains ordered; links run after all formulas, so a later formula sees an earlier target's raw predictor.

## Choosing a link

```python
sup.LinkFunction(bounds=(0.2, 4.0))       # scaled sigmoid, the class default
sup.LinkFunction()                       # scaled sigmoid into (0, 1)
sup.LinkFunction("softplus")             # positive, no finite upper bound
sup.LinkFunction("exp")                  # positive exponential
sup.LinkFunction("identity")             # unconstrained
sup.LinkFunction(lambda x: custom(x))     # custom NumPy function
```

Bounds must be finite and strictly increasing. Custom links must preserve array shape and return finite values. The scaled-sigmoid implementation is stable for large predictors; floating-point rounding can reach an interval endpoint. Choose a strictly positive lower bound when the simulator requires it.

Assign links per cognitive parameter using `link_function={"a": ..., "v": ...}`. A single `LinkFunction` or callable can be supplied to apply the same function to all simulator parameters. Omitted mapping entries, and `link_function=None`, use identity. Model does not guess domains from parameter names. Explicitly configured links also apply to simulator default parameters and parameter values bound from context, but do not transform context passed through the simulator's `context` argument.

The intercept is now on the predictor scale: at zero covariate, the cognitive parameter is `link_function(a_0)`. Coefficient priors should be chosen on this scale.

## Prior diagnostics and inference

```python
draws = model.sample_prior(batch_size=20, num_steps=3)
raw_coefficients = draws["local_params"]
cognitive_parameters = draws["model_params"]  # shape (batch, steps)

model.plot_joint_prior(num_steps=3)
model.plot_time_varying_prior(num_steps=3)
model.plot_time_invariant_prior(num_steps=3)
```

Prior plotting methods belong to `Model` and show raw prior quantities without resolving formulas, context, or links. For `v_diff = v_diff_0 + b_difficulty * difficulty`, they show `v_diff_0` and `b_difficulty`, not the resulting `v_diff`. Local trajectories, shared coefficients, and inferred transition hyperparameters are included. `plot_joint_prior()` additionally shows deterministic trajectories reconstructed from their curve coefficients. Fixed parameters, simulator defaults, simulator context, and formula-derived targets are excluded. Deterministic curve coefficients are shown when they have priors. Inferred contamination parameters are included; non-inferred contamination probabilities are excluded.

These plots sample inference priors without generating context, evaluating formulas, applying links, or calling the simulator. `sample_prior` remains available to inspect the linked cognitive parameters under supplied context, separately from inference-prior plotting.

`Model.sample` still returns raw inference groups: local and deterministic coefficient trajectories, shared coefficients, transition hyperparameters, and optionally fixed coefficients. New formula targets consumed by the simulator, such as `v_diff`, are also returned by `Model.sample` at their final linked values with shape `(batch_size, num_steps, 1)` for scalar parameters. `model.formula_keys` identifies these diagnostic fields. They are excluded from inference targets, summary inputs, and prior plots. Existing raw parameter, observation, and context names retain their original values. Resimulation should use raw coefficients, not these derived fields. `sample_prior` also exposes linked cognitive parameters without running the observation simulator.

```python
prediction = workflow.resimulate(
    posterior,
    num_sims=100,
    context={"difficulty": observed_difficulty},  # (datasets, steps)
    num_steps=3,
)
```

Resimulation selects original context with `data_idx` and repeats it for posterior draws. Supply original context to condition predictions on the observed design. If omitted, Model uses its configured context source. `num_steps` can be supplied for shared-only posteriors without a trial axis; explicit context can also provide the trial count. Fixed regression coefficients remain available during reconstruction.

For a transition-based `RandomChoiceContamination`, configure its probability link in Model, for example `link_function={"p_contaminated": sup.LinkFunction()}`. Contamination receives linked probabilities while inferred targets retain their raw scale. Direct augmentation calls without Model require probabilities already in [0, 1].

## Migrating existing models

This is a breaking modeling change:

- Remove `bounds` from transition constructors, including mixtures. Transitions always return raw trajectories; deterministic transitions no longer clip.
- Move intervals into `Model(link_function={name: LinkFunction(bounds=interval)})`. For regression, use the final cognitive parameter name, not an intercept or slope name.
- Replace `prior.plot_*prior(...)` with `model.plot_*prior(...)`.
- Replace `context_mapping=ContextMapping(...)` with `design_context=(...)` and `simulator_context=(...)` directly on Model. Former `formula_context` becomes `design_context`.
- Replace `design_matrix=DesignMatrix({"a": {"1": "a_0", "x": "b_a"}})` with `formula=Formula(["a = a_0 + b_a * x"])`; DesignMatrix and ContextMapping are removed from the public API.
- Remove use of `DEFAULT_BOUNDS`; interval defaults now belong to `LinkFunction`.
- Revisit coefficient priors and retrain existing estimators. Previously returned bounded trajectories and newly returned raw trajectories are different inference targets. Prior and posterior coefficient diagnostics use the raw inference scale.

`Prior.scale_factor` and `Prior.shift` remain distribution specifications: they define the sampled coefficient distribution. Cognitive-domain constraint transformations occur only in Model.


### Contamination probabilities

The probability belongs in `JointPrior`; the contamination process controls
whether it is inferred. The default is `infer=False`:

```python
prior = sup.JointPrior(
    a_0=sup.Prior("normal"),
    b_a=sup.Prior("normal"),
    p_contaminated=sup.Prior("beta", a=2, b=8),
)
contamination = sup.simulation.RandomChoiceContamination()  # infer=False
```

Pass both to `Model`. A scalar probability stays fixed; a `Prior` is shared
per dataset, and transitions support trial-varying probabilities. For raw
transition trajectories, use `link_function={"p_contaminated": sup.LinkFunction()}`.
Model samples the probability once and applies its link once before contamination.
If omitted, Model adds the default beta prior to a copy of the supplied JointPrior.

With `infer=True`, sampled probabilities and transition hyperparameters are
inference targets and appear in prior plots on their raw scale. With `infer=False`,
they are excluded from inference targets and prior plots; `Model.sample` still
returns the effective probability as augmentation metadata. Posterior resimulation
uses posterior probabilities in the first case and fresh prior draws in the second.


### Missingness probabilities

Specify `p_missing` in `JointPrior`, using a scalar, a `Prior`, or a transition.
`RandomMissingProcess(missing_value=-1)` owns only masking behavior. The probability
and its sampled hyperparameters are always nuisance quantities: they are excluded
from inference targets and prior plots. Model supplies the default beta prior when
omitted, and applies explicit `p_missing` links before masking.

Training draws from `Model.sample` apply missingness by default. Predictive
observations are complete by default:

```python
fig = model.plot_push_forward()                         # complete observations
fig = model.plot_push_forward(apply_missing=True)       # configured missingness
prediction = workflow.resimulate(estimates)             # complete observations
prediction = workflow.resimulate(estimates, apply_missing=True)
```

Opting in during posterior resimulation draws fresh missingness probabilities from
the prior; posterior estimates never provide `p_missing`. Contamination remains
applied in either mode. Direct missing-process calls require `probability=...`.
