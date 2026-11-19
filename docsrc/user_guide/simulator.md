# Simulator

The first step in our workflow is to formalize a data-generating process, which in Superstats we also call low-level observation model $\mathcal{G}$. Formally, it implements

$$x_t = \mathcal{G}(x_{1:t-1}; \theta_t, \lambda), \qquad t = 1, \dots, T$$

where $x_{1:t-1}$ is the observation history, $\theta_t$ collects the time-varying model parameters at time step $t$, and $\lambda$ collects time-invariant observation-model parameters. Any simulator randomness, such as diffusion noise, is implicit in $\mathcal{G}$. Superstats relies entirely on amortized Bayesian inference for estimating parameters $\theta$, and therefore only requires that $\mathcal{G}$ can be *simulated* — no closed-form likelihood is needed.

**Function signature.**
Superstats expects a function that:

- takes each observation model parameter as a keyword argument, passed as an array of shape `(num_steps,)`, one value per time step. Time-invariant parameters are tiled internally to `num_steps` before the call, so every parameter arrives with the same shape regardless of whether it was declared as time-varying, time-invariant, or fixed.
- returns a dict mapping observation names to arrays of shape `(num_steps,)`, i.e. one named observed variable per time step.

```python
def observation_model(
    param_1: np.ndarray,    # shape (num_steps,)
    param_2: np.ndarray,    # shape (num_steps,)
    param_n: np.ndarray,    # shape (num_steps,)
) -> dict[str, np.ndarray]: # variables with shape (num_steps,)
    ...
```

**Simulation speed.**
You will probably need to simulate many datasets, both for neural network training and for model verification. We recommend using just-in-time compilation and parallelizing across simulated datasets, for example via `numba` or `jax`, to substantially speed up the simulator's execution.

**Fixing parameters.**
We can fix parameters in two ways: either directly in the simulator, or in the next step, where we specify priors for the simulator. We recommend fixing a parameter in the simulator only if it is unlikely to be a target of inference, even though it could in theory be estimated (e.g., diffusion noise in an evidence accumulation model).

## Example: Diffusion Decision Model (DDM)

The diffusion decision model (DDM; [Ratcliff, 1978](https://doi.org/10.1037/0033-295X.85.2.59)) describes binary decisions as noisy evidence accumulation toward one of two boundaries:

$$dx = v_t \, dt + \sigma \, dW_t.$$

Evidence $x$ starts at $\text{bias}_t \cdot a_t$ and accumulates until it hits $a_t$ (upper boundary, choice $=1$) or $0$ (lower boundary, choice $=0$); the response time is $\tau_t$ (non-decision time) plus the time to reach a boundary. $v_t$ is the drift rate, $a_t$ the boundary separation (speed–accuracy trade-off), and $\text{bias}_t \in (0,1)$ the relative starting point — $0.5$ is unbiased, and values above or below shift the start point toward the upper or lower boundary, respectively.

The DDM is already implemented in Superstats.

```python
import superstats as sup

simulator = sup.simulation.cognitive.sample_ddm
```

`sample_ddm` integrates the DDM via Euler–Maruyama: at each step of size `dt`, it adds drift `v_t * dt` plus Gaussian noise scaled by `sigma * sqrt(dt)`, checking for a boundary crossing after each step. Trials that do not resolve within `max_steps` are marked as timeouts (RT $= -1.0$). Trials are simulated in parallel via `numba`.

```python
import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def sample_ddm(
    v: np.ndarray,
    a: np.ndarray,
    tau: np.ndarray,
    bias: np.ndarray,
    sigma: float = 1.0,
    dt: float = 0.001,
    max_steps: int = 10000,
) -> dict[str, np.ndarray]:
    num_steps = v.shape[0]
    response_time = np.empty(num_steps, dtype=np.float32)
    choice = np.empty(num_steps, dtype=np.float32)
    noise_scale = sigma * np.sqrt(dt)

    for i in prange(num_steps):
        v_t = v[i]
        a_t = a[i]
        t = tau[i]
        x = bias[i] * a_t
        drift_dt = v_t * dt

        for step in range(max_steps):
            t += dt
            x += drift_dt + noise_scale * np.random.normal()
            if x >= a_t:
                response_time[i] = t
                choice[i] = 1.0
                break
            if x <= 0.0:
                response_time[i] = t
                choice[i] = 0.0
                break
        else:
            response_time[i] = -1.0
            choice[i] = -1.0

    return {"response_time": response_time, "choice": choice}
```

## Other Built-in Simulators

Besides the DDM, Superstats has the following models implemented:

- `sup.simulation.cognitive.sample_rdm` $-$ Racing Diffusion Model ([Tillman, et al., 2020](https://doi.org/10.3758/s13423-020-01719-6))
- `sup.simulation.cognitive.sample_cdm` $-$ Circular Diffusion Model ([Smith, 2016](https://doi.org/10.1037/rev0000023))

Please feel free to contribute additional models, either by opening a pull request or an issue with a feature request.
