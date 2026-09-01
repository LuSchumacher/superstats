# Superstats

Superstats is a Python library for simulation and Bayesian estimation of dynamic models with time-varying parameters.

The library aims to be domain-agnostic, but for now focuses on cognitive modeling. It provides users with:

- A lean API for non-stationary models: specify which parameters change across time, and how.
- A library of transition models: random walks, ARs, levy flights, jump processes, mixtures, and Gaussian processes.
- Built-in cognitive models, plus a plug-in interface for any simulator of your own.
- Amortized Bayesian inference built on top of [BayesFlow](https://github.com/bayesflow-org/bayesflow): train once, then quickly fit every data set.
- Diagnostics and visualization tools for every critical step in a principled Bayesian workflow.

## Conceptual overview

:::{div} conceptual-overview
:::{image} /_static/superstats-arch-light.svg
:alt: Overview graphic of superstats
:::
:::


A superstatistical model has two levels. A **low-level observation model** $\mathcal{G}$ generates the data at each time step.
A **high-level transition model** $\mathcal{T}$ describes how the parameters of that model evolve:

$$\theta_t = \mathcal{T}(\theta_{0:t-1}; \eta) \qquad x_t = \mathcal{G}(x_{1:t-1}; \theta_t, \lambda)$$

Superstats trains a neural estimator on simulations from any generative model of this form and returns the joint posterior $p(\theta_{1:T}, \eta, \lambda \mid x_{1:T})$ over all time-varying parameters $\theta_{1:T}$ and time-invariant parameters $(\eta, \lambda)$.


## Install

We support Python 3.12 and 3.13. Install the latest release from PyPI:

```bash
pip install superstats
```

If you want the latest features, you can install from source:

```bash
pip install git+https://github.com/LuSchumacher/superstats.git@dev
```

### Deep learning backend

By default, `superstats` installs [JAX](https://docs.jax.dev/en/latest/installation.html) on Linux and macOS, and [PyTorch](https://pytorch.org/get-started/locally/) on Windows. This is because JAX does not natively support GPU acceleration on Windows. You can also manually install and configure any of the three backends:

- [Install JAX](https://jax.readthedocs.io/en/latest/installation.html)
- [Install PyTorch](https://pytorch.org/get-started/locally/)
- [Install TensorFlow](https://www.tensorflow.org/install)


## Getting started

A workflow using a diffusion decision model whose drift rate and threshold can vary over time:

```python
import superstats as sup

# 1. Assume which parameters move, and how
joint_prior = sup.JointPrior(
    v=sup.transition.RandomWalk(), a=sup.transition.RandomWalk(), tau=sup.Prior(dist="halfnormal", scale=0.15), bias=0.5
)

# 2. Plug in an observation model (any simulator will do)
model = sup.Model(
    prior=joint_prior, simulator=sup.simulation.sample_ddm, missing="random", contamination="random_choice"
)

# 3. Train a neural approximator
workflow = sup.Workflow(model=model)
history = workflow.fit_online(num_steps=100, epochs=20, batch_size=16)

# 4. Fit any number of data sets, instantly
samples = workflow.sample(data=rt_data, num_samples=250)
```

A GPU is highly recommended for training and inference. Start with the [user-guide quickstart](user_guide/quickstart.ipynb), then explore the [examples folder](https://github.com/LuSchumacher/superstats/tree/main/examples), including the [minimal workflow demo](https://github.com/LuSchumacher/superstats/blob/main/examples/minimal_workflow_demo.ipynb), for a complete analysis with training and diagnostics.


## Contributing

Contributions are welcome. Install from source and see [CONTRIBUTING.md](contributing.md) for details.


## Reporting issues

Please open an issue on [GitHub](https://github.com/LuSchumacher/superstats/issues) for bug reports and
feature requests. For questions about the underlying inference machinery, the
[BayesFlow Forums](https://discuss.bayesflow.org/) are a good place to ask.


## Citation

If you use Superstats in your research, please cite:

```bibtex
@article{schumacher2023neural,
  title   = {Neural superstatistics for {B}ayesian estimation of dynamic cognitive models},
  author  = {Schumacher, Lukas and B{\"u}rkner, Paul-Christian and Voss, Andreas and K{\"o}the, Ullrich and Radev, Stefan T.},
  journal = {Scientific Reports},
  volume  = {13},
  number  = {1},
  pages   = {13778},
  year    = {2023},
  doi     = {10.1038/s41598-023-40278-3}
}

@article{schumacher2025validation,
  title   = {Validation and comparison of non-stationary cognitive models: A diffusion model application},
  author  = {Schumacher, Lukas and Schnuerch, Martin and Voss, Andreas and Radev, Stefan T.},
  journal = {Computational Brain \& Behavior},
  volume  = {8},
  number  = {2},
  pages   = {191--210},
  year    = {2025},
  doi     = {10.1007/s42113-024-00218-4}
}
```

## License

MIT

:::{toctree}
:maxdepth: 2
:hidden:

user_guide/index
examples/index
api/index
contributing
:::
