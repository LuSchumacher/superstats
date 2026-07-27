import os
import sys

import numpy as np
import pytest

os.environ.setdefault("KERAS_BACKEND", "torch" if sys.platform == "win32" else "jax")


@pytest.fixture(autouse=True)
def _seed_numpy():
    """Seed numpy's global RNG before every test for reproducibility."""
    np.random.seed(0)
