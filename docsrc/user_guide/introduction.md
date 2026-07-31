# Introduction

In Superstats, a dynamic model is built from two pieces: a **low-level observation model** (e.g., a cognitive model such as the Diffusion Decision Model) that generates data at each time step, and a **high-level transition model** that describes how the model's parameters evolve over time.

A typical amortized Bayesian workflow ([Li et al., 2026](https://openreview.net/forum?id=osV7adJlKD)) consists of the following steps:

1. **Define the observation model** as a data simulator.
2. **Specify a joint prior**, assigning a transition model to each parameter that should vary over time, and a standard prior to those that should not.
3. **Prior push-forward checks:** Simulate from the model and ask whether the implied parameter trajectories and data are consistent with your domain knowledge. Adjust the priors and transition models until they are.
4. **Set up the amortized Bayesian workflow:** specify a neural approximator consisting of a summary and inference network.
5. **Train the neural approximator** on simulations from the model.
6. **Model verification:** Check that the approximate posteriors are well calibrated (via simulation-based calibration) and that the model and design can answer your question at all (via parameter recovery and posterior contraction). If they cannot, return to steps 1–2 and revise.
7. **Fit empirical data** for any number of datasets, at negligible cost.
8. **Evaluate the absolute model fit:** Re-simulate data from the posterior and ask whether the model reproduces the patterns you care about. A model that misses them is not worth interpreting, no matter how well it did in step 6.
9. **Inspect the posteriors** of the time-varying and time-invariant parameters.
