import numpy as np
from numba import njit

@njit
def sample_ddm(
    v: float,
    a: float,
    tau: float,
    bias: float,
    sigma: float = 1.0,
    dt: float = 0.001,
    max_steps: int = 10000,
):
    c = sigma * np.sqrt(dt)
    x = bias * a
    t = tau
    for _ in range(max_steps):
        t += dt
        x += v * dt + c * np.random.normal()
        if x >= a:
            return np.array([t, 1.0], dtype=np.float32)
        if x <= -a:
            return np.array([t, 0.0], dtype=np.float32)
    # No decision within max_steps
    return np.array([-1.0, -1.0], dtype=np.float32)