# User guide

This guide takes you from a simulator to posterior estimates for models with time-varying parameters.

## Choose a route

| If you want to... | Start here |
|---|---|
| Understand the modeling assumptions | [Core concepts](introduction.md) |
| Connect your own data-generating process | [Simulator](simulator.md) |
| Decide which parameters vary over time | [Joint prior](joint_prior.md) |
| Check what the complete model generates | [Model](model.md) |
| Add experimental designs and simulator inputs | [Context](context.md) |
| Represent missing or contaminated observations | [Data augmentation](augmentation.md) |
| Estimate varying and invariant parameters together | [Posterior approximators](approximators.md) |
| Check the model and posterior approximation | [Diagnostics](diagnostics.md) |

For a compact end-to-end demo notebook, see the [examples folder](https://github.com/LuSchumacher/superstats/tree/main/examples).
For individual classes and function signatures, use the [API reference](../api/index.rst).

```{toctree}
:maxdepth: 1
:titlesonly:
:numbered:

introduction
simulator.md
joint_prior.md
model.md
context.md
augmentation.md
approximators
diagnostics.md
```
