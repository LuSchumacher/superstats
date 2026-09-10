# Core concepts

Superstats estimates models whose parameters may change across ordered observations.
A model has two layers:

- An **observation model** $\mathcal{G}$ generates an observation $x_t$ from parameters $\theta_t$ at step $t$.
- A **transition model** $\mathcal{T}$ describes how those parameters evolve across steps.

In compact form,

$$
\theta_t = \mathcal{T}(\theta_{0:t-1}; \eta),
\qquad
x_t = \mathcal{G}(\theta_t; \lambda),
$$

where $\eta$ contains transition hyperparameters and $\lambda$ contains time-invariant observation-model parameters.
The observation model only needs to be simulatable; Superstats does not require a tractable likelihood.

## Why amortized inference?

Superstats uses amortized Bayesian inference through [BayesFlow](https://github.com/bayesflow-org/bayesflow).
Instead of fitting one dataset with a new optimization or sampling run, it trains a neural posterior approximator on many simulated parameter-data pairs.
Training has an upfront cost, but the trained approximator can then return posterior draws for many datasets quickly.

The target is the joint posterior

$$
p(\theta_{1:T}, \eta, \lambda \mid x_{1:T}).
$$

Because this posterior is learned from simulations, it is only trustworthy in regions represented by the prior-predictive training distribution.
Empirical data that look unlike the simulations are out of distribution, even if their array shapes are valid.

## A principled workflow

1. **Define the observation model.** Choose a built-in simulator or implement one with named array outputs.
2. **Specify the joint prior.** Decide which parameters vary, what dynamics are plausible, and which values are fixed or shared.
3. **Check prior trajectories and simulated data.** Revise assumptions until trajectories and observations are scientifically credible.
4. **Train the approximator.** Use online training for fresh simulations per batch or offline training for a fixed, reusable simulation set.
5. **Verify on new simulations.** Inspect parameter recovery, posterior contraction, and simulation-based calibration. If the model cannot recover known simulated values, do not interpret an empirical fit.
6. **Fit empirical data.** Preserve observation order and use the same coding, sequence length, and missing-data convention used during training.
7. **Run posterior re-simulations.** Re-simulate from posterior draws and compare distributions and temporal patterns with the observed data.
8. **Interpret the posterior.** Inspect time-varying trajectories together with uncertainty and the time-invariant transition parameters that regularize them.

Steps 3 and 5 are decision points, not formalities.
Implausible simulations call for revised priors or model assumptions; poor recovery or calibration calls for a revised model, design, training budget, or network.

## Data and shape vocabulary

After a `Model` wraps the simulator, its output is a dictionary. Observation arrays have shape `(batch_size, num_steps)`.
Inference targets have a trailing component dimension and are either local, with shape `(batch_size, num_steps, 1)`, or time-invariant, with shape `(batch_size, 1)`.
Training may tile time-invariant targets across steps for the default recurrent network; this is an alignment detail and does not make them time-varying.
