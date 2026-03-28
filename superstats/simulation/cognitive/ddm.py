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
) -> np.ndarray:
    
    num_steps = v.shape[0]
    data = np.empty((num_steps, 2), dtype=np.float32)
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
                data[i, 0] = t
                data[i, 1] = 1.0
                break
            if x <= -a_t:
                data[i, 0] = t
                data[i, 1] = 0.0
                break
        else:
            data[i, 0] = -1.0
            data[i, 1] = -1.0

    return data
