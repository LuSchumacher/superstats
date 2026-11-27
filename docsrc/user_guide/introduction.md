# Core concepts

Superstats estimates models whose parameters may change across ordered observations.
A model has two layers:

- An **observation model** $\mathcal{G}$ generates an observation $x_t$ from the observation history $x_{1:t-1}$ and parameters $\theta_t$ at step $t$.
- A **transition model** $\mathcal{T}$ describes how those parameters evolve across steps.

In compact form,

$$
\theta_t = \mathcal{T}(\theta_{0:t-1}; \eta),
\qquad
x_t = \mathcal{G}(x_{1:t-1}; \theta_t, \lambda),
$$

where $\eta$ contains transition hyperparameters and $\lambda$ contains time-invariant observation-model parameters.
Let $\phi = (\eta, \lambda)$ collect these time-invariant parameters.

## Why amortized inference?

Superstats relies entirely on amortized Bayesian inference through [BayesFlow](https://github.com/bayesflow-org/bayesflow) for parameter estimation, and therefore only requires that $\mathcal{G}$ can be *simulated* — no closed-form likelihood is needed.
The ability to estimate models with intractable likelihoods is a major advantage of this approach.

Instead of fitting one dataset with a new optimization or sampling run, it trains a neural posterior estimator on many simulated parameter-data pairs.
Training has an upfront cost, but the trained estimator can then return posterior draws for many datasets quickly.

Superstats supports four posterior approximations for the time-varying trajectory $\theta_{1:T}$ and the invariant parameters $\phi$:

- **Marginal filtering (MF):** $q_{\mathrm{MF}}(\theta_{1:T}, \phi \mid x_{1:T}) = q(\phi \mid x_{1:T}) \prod_{t=1}^T q(\theta_t \mid \phi, x_{1:t})$.
- **Marginal smoothing (MS):** $q_{\mathrm{MS}}(\theta_{1:T}, \phi \mid x_{1:T}) = q(\phi \mid x_{1:T}) \prod_{t=1}^T q(\theta_t \mid \phi, x_{1:T})$.
- **Full joint filtering (JF):** $q_{\mathrm{JF}}(\theta_{1:T}, \phi \mid x_{1:T}) = q(\phi \mid x_{1:T}) \prod_{t=1}^T q(\theta_t \mid \theta_{1:t-1}, \phi, x_{1:t})$.
- **Full joint smoothing (JS):** $q_{\mathrm{JS}}(\theta_{1:T}, \phi \mid x_{1:T}) = q(\phi \mid x_{1:T}) \prod_{t=1}^T q(\theta_t \mid \theta_{1:t-1}, \phi, x_{1:T})$.

Filtering factors use observations up to the current step, while smoothing factors use the full observation sequence.
Joint factors also condition on the preceding parameter trajectory.
See the [estimator guide](estimators.md) for configuration and tensor shapes.

Because these posterior approximations are learned from simulations, they are only trustworthy in regions represented by the prior-predictive training distribution.
Empirical data that look unlike the simulations are out of distribution, even if their array shapes are valid.

## A principled workflow

1. **Define the observation model.** Choose a built-in model simulator or implement one on your own.
2. **Specify the joint prior.** Decide which parameters vary, what dynamics are plausible, and which values are fixed or shared.
3. **Check prior trajectories and simulated data.** Revise assumptions until trajectories and observations are scientifically reasonable.
4. **Train the estimator.** Use online training for fresh simulations per batch or offline training for a fixed, reusable simulation set.
5. **Verify on new simulations.** Inspect parameter recovery, posterior contraction, and simulation-based calibration. If the model cannot recover known simulated values, do not interpret an empirical fit.
6. **Fit empirical data.** Preserve observation order and use the same coding, sequence length, and missing-data convention used during training.
7. **Run posterior re-simulations.** Re-simulate from posterior draws and compare distributions and temporal patterns with the observed data.
8. **Interpret the posterior.** Inspect time-varying trajectories together with uncertainty and the time-invariant transition parameters that regularize them.

Steps 3 and 5 are decision points, not formalities.
Implausible simulations call for revised priors or model assumptions; poor recovery or calibration calls for a revised model, design, training budget, or network.

## Data and shape vocabulary

After a `Model` wraps the simulator, its output is a dictionary. Observation arrays have shape `(batch_size, num_steps)`.
Inference targets have a trailing component dimension and are either time-varying, with shape `(batch_size, num_steps, 1)`, or time-invariant, with shape `(batch_size, 1)`.
