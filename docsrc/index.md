# Superstats

**superstats** is a Python library for simulation and Bayesian estimation of models with time-varying parameters.

The library is general, but for now focuses on cognitive modeling. It provides users with:

- A lean API for non-stationary models: specify which parameters change across time, and how.
- A library of transition models: random walks, Ornstein–Uhlenbeck, autoregression, jump processes, mixtures, and Gaussian processes.
- Built-in cognitive models, plus a plug-in interface for any simulator of your own.
- Amortized Bayesian inference build on top of [BayesFlow](https://github.com/bayesflow-org/bayesflow): train once, then quickly fit every data set.
- Diagnostic and visualization tools for every critical step in a principled Bayesian workflow.

## Conceptual overview

A superstatistical model has two levels. A **low-level observation model** $\mathcal{G}$ generates the data at each time step.
A **high-level transition model** $\mathcal{T}$ describes how the parameters of that model evolve:

$$\theta_t = \mathcal{T}(\theta_{0:t-1}; \eta) \qquad x_t = \mathcal{G}(x_{1:t-1}; \theta_t, \lambda)$$

`superstats` trains a neural estimator on simulations from any generative model of this form and returns the joint posterior $p(\theta_{1:T}, \eta, \lambda \mid x_{1:T})$ over all time-varying parameters $\theta_{1:T}$ and time-invariant parameters $(\eta, \lambda)$.


## Install

We support Python 3.11 to 3.13. Install the latest version from source:

```bash
pip install superstats
```

If you want the latest features, you can install from source:

```bash
pip install git+https://github.com/LuSchumacher/superstats.git@dev"
```

## Getting started

A complete workflow: a diffusion decision model whose drift rate and thresold are free to vary across time:

```pythons
import superstats as sup

# 1. Assume which parameters move, and how
joint_prior = sup.JointPrior(
    v=sup.transition.RandomWalk(),
    a=sup.transition.RandomWalk(),
    tau=sup.Prior(dist="halfnormal", scale=0.15),
    bias=0.5
)

# 2. Plug in an observation model (any simulator will do)
generative_model = sup.GenerativeModel(
    prior=joint_prior,
    model=sup.simulation.sample_ddm,
    missing="random",
    contamination="random_choice"
)

# 3. Train a neural approximator
workflow = sup.Workflow(simulator=generative_model)
history = workflow.fit_online(num_steps=100, epochs=20, batch_size=16)

# 4. Fit any number of data sets, instantly
samples = workflow.sample(data=rt_data, num_samples=250)
```

It is highly recommended to use a GPU for fast training and inference. For an in-depth exposition, check out the examples below.

## Examples

| Notebook | What it covers |
|---|---|
| [Minimal workflow demo](examples/minimal_workflow_demo.ipynb) | Short path from prior to posterior |
| [Extensive workflow demo](examples/extensive_workflow_demo.ipynb) | More indepth workflow, end to end, with all diagnostics |

More examples are always welcome — if you have an application, please consider opening a pull request.


## Contributing

Contributions are welcome. Install from source and see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## Reporting issues

Please open an issue on [GitHub](https://github.com/LuSchumacher/superstats/issues) for bug reports and
feature requests. For questions about the underlying inference machinery, the
[BayesFlow Forums](https://discuss.bayesflow.org/) are a good place to ask.

## Citation

If you use `superstats` in your research, please cite:

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

```{toctree}
:maxdepth: 2
:caption: Contents

getting_started
user_guide/index
examples/index
api/index
contributing
```
