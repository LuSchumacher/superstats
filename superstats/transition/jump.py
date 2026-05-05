from typing import Tuple, Dict, Any
import numpy as np
from numba import njit, prange

from .transition import Transition, Prior
from superstats.utils.transformations import scaled_sigmoid


@njit(parallel=True, fastmath=True)
def _sample_jump_process(
    local_params: np.ndarray,
    p_jump: np.ndarray,
    proposals: np.ndarray,
    bounds: np.ndarray,
) -> np.ndarray:

    batch_size, steps = local_params.shape
    lower, upper = bounds[0], bounds[1]

    uniform = np.random.rand(batch_size, steps - 1)

    for b in prange(batch_size):
        for t in range(1, steps):
            if uniform[b, t - 1] < p_jump[b]:
                local_params[b, t] = proposals[b, t - 1]
            else:
                local_params[b, t] = local_params[b, t - 1]

        local_params[b, :] = scaled_sigmoid(local_params[b, :], lower, upper)

    return local_params


class Jump(Transition):
    """
    Simple jump process:
    stay or jump to proposal draw.
    """

    def __init__(
        self,
        bounds: Tuple[float, float] | None = None,
        initial_prior: Prior | None = None,
        p_jump: float | Prior = 1.0,
        proposal_prior: Prior | None = None,
    ):
        super().__init__(bounds, initial_prior)

        self.hyper_specs = {
            "p_jump": p_jump,
        }

        self.proposal_prior = proposal_prior or Prior("normal", loc=0.0, scale=1.0)
        self.transition_type = "jump"

    def sample(self, batch_size: int, steps: int) -> Dict[str, Any]:

        local_params = np.empty((batch_size, steps), dtype=self.dtype)
        local_params[:, 0] = self.initial_prior.sample(batch_size).astype(self.dtype)

        hyper, fixed = self._resolve_hyperparams(batch_size)

        if "p_jump" in hyper:
            p_jump = hyper["p_jump"]
        else:
            p_jump = np.full(batch_size, fixed["p_jump"], dtype=self.dtype)

        proposals = self.proposal_prior.sample(batch_size * (steps - 1)).reshape(
            batch_size,
            steps - 1,
        ).astype(self.dtype)

        local_params = _sample_jump_process(
            local_params,
            p_jump,
            proposals,
            self.bounds,
        )

        return {
            "local_params": local_params,
            "hyper_params": hyper,
            "fixed_params": fixed,
        }

    def sample_one_step(self, x: np.ndarray, params: Dict[str, Any]) -> np.ndarray:

        p_jump = np.asarray(params["p_jump"], dtype=self.dtype)
        proposal = self.proposal_prior.sample(x.shape[0]).astype(self.dtype)

        jump = np.random.rand(x.shape[0]) < p_jump
        return np.where(jump, proposal, x)