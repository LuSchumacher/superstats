from typing import Tuple, Dict, Any
import numpy as np
from numba import njit, prange

from .transition import Transition
from superstats.prior import Prior
from superstats.utils.transformations import scaled_sigmoid


@njit(parallel=True, fastmath=True)
def _sample_jump_ar1(
    local_params: np.ndarray,
    sigma: np.ndarray,
    phi: np.ndarray,
    delta: np.ndarray,
    p_jump: np.ndarray,
    bounds: Tuple[float, float],
) -> np.ndarray:
    batch_size, steps = local_params.shape
    lower, upper = bounds

    noise = np.random.randn(batch_size, steps - 1)
    jump_draws = np.random.rand(batch_size, steps - 1)

    for b in prange(batch_size):
        x_prev = local_params[b, 0]

        for t in range(1, steps):
            if jump_draws[b, t - 1] < p_jump[b]:
                x_prev = np.random.rand()
            else:
                x_prev = phi[b] * x_prev + delta[b] + sigma[b] * noise[b, t - 1]
                local_params[b, t] = x_prev

        local_params[b, :] = scaled_sigmoid( local_params[b, :], lower, upper)

    return local_params


@njit(parallel=True, fastmath=True)
def _sample_jump_rw(
    local_params: np.ndarray,
    sigma: np.ndarray,
    delta: np.ndarray,
    p_jump: np.ndarray,
    bounds: Tuple[float, float],
) -> np.ndarray:
    batch_size, steps = local_params.shape
    lower, upper = bounds

    noise = np.random.randn(batch_size, steps - 1)
    jump_draws = np.random.rand(batch_size, steps - 1)

    for b in prange(batch_size):
        x_prev = local_params[b, 0]
        
        for t in range(1, steps):
            if jump_draws[b, t - 1] < p_jump[b]:
                x_prev = np.random.rand()
            else:
                x_prev = x_prev + delta[b] + sigma[b] * noise[b, t - 1]
            
            local_params[b, t] = x_prev

        local_params[b, :] = scaled_sigmoid(local_params[b, :], lower, upper)

    return local_params


@njit(parallel=True, fastmath=True)
def _sample_jump_ou(
    local_params: np.ndarray,
    sigma: np.ndarray,
    theta: np.ndarray,
    mu: np.ndarray,
    p_jump: np.ndarray,
    bounds: Tuple[float, float],
) -> np.ndarray:
    batch_size, steps = local_params.shape
    lower, upper = bounds

    noise = np.random.randn(batch_size, steps - 1)
    jump_draws = np.random.rand(batch_size, steps - 1)

    for b in prange(batch_size):
        x_prev = local_params[b, 0]
        
        for t in range(1, steps):
            if jump_draws[b, t - 1] < p_jump[b]:
                x_prev = np.random.rand()
            else:
                x_prev = (
                    x_prev
                    + theta[b] * (mu[b] - x_prev)
                    + sigma[b] * noise[b, t - 1]
                )
            
            local_params[b, t] = x_prev

        local_params[b, :] = scaled_sigmoid(local_params[b, :], lower, upper)

    return local_params


class JumpTransition(Transition):
    """
    Jump-augmented transition.

    At each step:
        with prob p_jump -> jump
        else -> base dynamics
    """

    def __init__(
        self,
        base_transition: Transition,
        jump_prob: Prior | float | None = None,
    ):
        super().__init__(
            bounds=base_transition.bounds,
            initial_prior=base_transition.initial_prior,
        )

        self.base_transition = base_transition
        self.transition_type = base_transition.transition_type

        self.global_params = dict(base_transition.global_params)
        self.global_params["jump_prob"] = jump_prob

        if self.transition_type == "ar1":
            self.kernel = _sample_jump_ar1
        elif self.transition_type == "rw":
            self.kernel = _sample_jump_rw
        elif self.transition_type == "ou":
            self.kernel = _sample_jump_ou
        else:
            raise ValueError(f"Unknown transition type: {self.transition_type}")

    def sample(
        self,
        batch_size: int,
        steps: int
    ) -> Dict[str, Any]:
        local_params = np.empty((batch_size, steps), dtype=self.dtype)
        local_params[:, 0] = self.initial_prior.sample(batch_size)
        global_params, infer_mask = self.sample_global_params(batch_size)

        local_params = self.kernel(
            local_params,
            *global_params.values(),
            self.bounds,
        )

        return {
            "local_params": local_params,
            "global_params": global_params,
            "infer_mask": infer_mask,
        }
