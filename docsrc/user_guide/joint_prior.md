# Joint Prior

```python
import superstats as sup
```

A `JointPrior` specifies how every observation-model parameter behaves and whether it is estimated.
Superstats groups the resulting values into five parameter types:

| Parameter type | Changes over steps? | Estimated? | Example |
|---|---:|---:|---|
| `local_param` | Yes | Yes | Trajectory produced by a `StochasticTransition` |
| `hyper_param` | No | Yes | Transition hyperprior such as `sigma` or `slope` |
| `deterministic_param` | Yes | Indirectly | Trajectory produced by a `DeterministicTransition` |
| `shared_param` | No | Yes | A `Prior` shared across all time steps |
| `fixed_param` | No | No | A fixed scalar |

The `JointPrior` collects these specifications into one coherent prior. Its keyword names must match the simulator's parameter names.

```python
joint_prior = sup.JointPrior(
    param_1=sup.transition.RandomWalk(...),  # stochastic transition
    param_2=sup.transition.Linear(...),      # deterministic transition
    param_3=sup.Prior(...),                  # shared parameter
    param_4=0.5,                             # fixed parameter
)
```

## Stochastic Transitions

Superstats implements the following stochastic transition models:

- `RandomWalk`: with hyperparameters `sigma` for the scale of Gaussian noise and `delta` for linear additive drift.
- `AutoRegression`: an AR(1) process with hyperparameters `sigma` for Gaussian noise, `phi` for the autoregressive coefficient, and `delta` for additive drift.
- `OrnsteinUhlenbeck`: a mean-reverting process with hyperparameters `sigma` for diffusion noise, `mu` for the long-run mean, and `theta` for the mean-reversion speed.
- `LevyFlight`: a random walk with alpha-stable noise, with hyperparameters `sigma` for noise scale, `delta` for additive drift, and `alpha` for tail heaviness/stability.
- `Jump`: a jump process with `p_jump` controlling the probability of jumping to a new proposal value at each step.
- `GaussianProcess`: trajectories sampled from a Gaussian process using configurable RBF, linear, periodic, or composite kernels.
- `Mixture`: a mixture of two or more stochastic transition models, with Dirichlet-distributed mixture weights (does not work with `GaussianProcess`).

For each hyperparameter, pass a `Prior` to estimate it or a scalar to fix it and exclude it from inference.
Omitting a hyperparameter uses the package default.
Inspect those defaults before relying on them, because they determine the trajectories represented during training.

The `initial_prior` argument specifies a prior or fixed scalar for the initial value in unconstrained space. Parameter bounds are configured on `Model` with a `LinkFunction`.

**Example.**


```python
prior = sup.JointPrior(
    param_1=sup.transition.RandomWalk(
        initial_prior=sup.Prior("normal", loc=0.0, scale=2.0),
        sigma=sup.Prior("halfnormal", scale=0.2),
        delta=0.0,
    ),
    param_2=sup.Prior("normal", loc=0.0, scale=1.0),
)
latent_link_function = {"param_1": sup.LinkFunction(bounds=(-6.0, 6.0))}
```

Here, `param_1` and `param_1_sigma` are estimated; `param_1_delta` is fixed.
The keyword names must match arguments in the observation-model simulator.

### Mixture

The `Mixture` transition allows mixing the following stochastic transition models:

- `RandomWalk`
- `AutoRegression`
- `OrnsteinUhlenbeck`
- `LevyFlight`
- `Jump`

The most interesting and sensible mixture is between one of the first four transitions and a `Jump`, since this lets a parameter follow smooth, gradual dynamics most of the time while occasionally undergoing a sudden discrete jump.

A few notes on using the `Mixture` transition:

- `initial_prior` must be defined once at initialization of the `Mixture` itself and must **not** be specified again within the individual transitions it contains. Bounds are configured once for the final parameter through `Model.latent_link_function`.

- When a `Jump` transition is included in the `Mixture`, its `p_jump` is automatically fixed to $1.0$, since the `mixture_weights` already govern the probability of a jump occurring at a given time step. It would not be sensible for the `Jump` component to be selected at a given step and then, due to `p_jump < 1.0`, have a chance of no jump actually occurring.

- As noted earlier, parameter trajectories are generated in an unconstrained space and then transformed by the parameter's model-level link. With a scaled-sigmoid link, setting `proposal_prior=sup.prior.Prior(dist="logistic", loc=0, scale=1)` for the `Jump` component results in a uniform distribution over the linked bounds.

**Example.**

```python
prior = sup.JointPrior(
    param_1=sup.transition.RandomWalk(
        initial_prior=sup.Prior("normal", loc=0, scale=2),
        sigma=sup.Prior("halfnormal", scale=0.2),
    ),
    param_2=sup.transition.Mixture(
        initial_prior=sup.Prior("normal", loc=0, scale=1),
        transitions=[
            sup.transition.OrnsteinUhlenbeck(
                sigma=sup.Prior("halfnormal", scale=0.1),
                theta=sup.Prior("halfnormal", scale=0.05),
            ),
            sup.transition.Jump(
                proposal_prior=sup.Prior("logistic", loc=0, scale=1),
            ),
        ],
        mixture_weights=sup.Prior("dirichlet", alpha=[9, 1]),
    ),
    param_3=0.5,
)
latent_link_function = {
    "param_1": sup.LinkFunction(bounds=(-6, 6)),
    "param_2": sup.LinkFunction(bounds=(0, 1)),
}
```

## Deterministic Transitions

Deterministic transitions represent structured change with a small set of curve parameters. Superstats provides `Linear`, `Polynomial`, `Exponential`, and `Logarithmic`. The trajectory itself is not a direct inference target; any curve parameter specified with a `Prior` becomes a time-invariant `hyper_param`, and the posterior curve is reconstructed from those values. A scalar curve parameter stays fixed.

By default, deterministic transitions evaluate time on a normalized interval from 0 to 1. This makes a coefficient describe change over the complete sequence rather than change per observation. Set `normalize_steps=False` only when coefficients should act on raw step indices. Use `LinkFunction("clip", bounds=...)` on `Model` when values outside an interval should be clipped.

```python
prior = sup.JointPrior(
    param_1=sup.transition.Linear(
        intercept=sup.Prior("normal", loc=1.5, scale=0.3),
        slope=sup.Prior("normal", loc=0.0, scale=0.4),
    ),
    param_2=sup.transition.Polynomial(
        intercept=0.0,
        betas=[
            sup.Prior("normal", loc=0.0, scale=0.5),
            sup.Prior("normal", loc=0.0, scale=0.2),
        ],
        degree=2,
    ),
)
latent_link_function = {
    "param_1": sup.LinkFunction("clip", bounds=(0.0, 4.0)),
    "param_2": sup.LinkFunction("clip", bounds=(-2.0, 2.0)),
}
```

Plot deterministic and stochastic trajectories together with `model.plot_time_varying_prior(...)` to check for implausible trends.

## Priors

For all hyperparameters and shared parameters, we can specify a standard prior via the `Prior` class.
Superstats implements the following distributions:

- **`normal`** — Gaussian, parameterized by `loc` (mean) and `scale` (standard deviation).
- **`halfnormal`** — folded Gaussian centered at 0, non-negative, parameterized by `scale`. Commonly used for scale/noise parameters (e.g., `sigma`).
- **`uniform`** — uniform over `[low, high]`.
- **`beta`** — Beta distribution on `[0, 1]`, parameterized by shape parameters `a` and `b`.
- **`logistic`** — logistic distribution, parameterized by `loc` and `scale`. Often used as a `proposal_prior` in `Jump` transitions, since a scaled-sigmoid model link maps it to a uniform distribution over the link bounds.
- **`dirichlet`** — distribution over the simplex, parameterized by a concentration vector `alpha`. Used, for example, to specify `mixture_weights` in a `Mixture` transition.

Every `Prior` additionally supports an optional linear transform of the drawn samples via `scale_factor` and additive `shift`, i.e., `scale_factor * samples + shift`.

Feel free to open an issue if a prior you'd like to use isn't implemented yet.
