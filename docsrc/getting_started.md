# Getting Started

Install superstats from PyPI.

Using `uv`:

```bash
uv add superstats
```

Or, usin goold old `pip`:

```bash
pip install superstats
```

Then create a Python file or a notebook with the following code:

```python
from superstats.prior import JointPrior, Prior
from superstats.transition import RandomWalk

prior = JointPrior(
    drift=RandomWalk(bounds=(-2.0, 2.0)),
    threshold=Prior("logistic", loc=1.0, scale=0.2),
)

samples = prior.sample(
    batch_size=32,
    num_steps=100,
)

print(samples)
```

This example creates a joint prior containing:

* A time-varying `drift` parameter modeled as a random walk.
* A time-invariant `threshold` parameter with a logistic prior.
* `32` simulated samples, each containing `100` time steps.

The returned `samples` dictionary can be passed to simulators, models, or plotting utilities.
