# User guide

This guide takes you from a simulator to posterior estimates for models with time-varying parameters.

## Choose a route

| If you want to... | Start here |
|---|---|
| Understand the modeling assumptions | [Core concepts](introduction.md) |
| Connect your own data-generating process | [Simulator](simulator.ipynb) |
| Decide which parameters vary over time | [Joint prior](joint_prior.ipynb) |
| Check what the complete model generates | [Model](model.ipynb) |
| Represent missing or contaminated observations | [Data augmentation](augmentation.ipynb) |
| More coming soon... |  |

For a compact end-to-end demo notebook, see the [examples folder](https://github.com/LuSchumacher/superstats/tree/main/examples).
For individual classes and function signatures, use the [API reference](../api/index.rst).

```{toctree}
:maxdepth: 1
:titlesonly:
:numbered:

introduction
simulator.ipynb
joint_prior.ipynb
model.ipynb
augmentation.ipynb
```
