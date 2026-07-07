import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _seed_numpy():
    """Seed numpy's global RNG before every test for reproducibility."""
    np.random.seed(0)
