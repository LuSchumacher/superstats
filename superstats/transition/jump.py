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
    bounds: Tuple[float, float],
) -> np.ndarray:
    # Sample a jump process where each step either stays or jumps to a proposal.
    batch_size, steps = local_params.shape
    lower, upper = bounds

    uniform = np.random.rand(batch_size, steps - 1)

    for b in prange(batch_size):
        for t in range(1, steps):
            if uniform[b, t - 1] < p_jump[b]:
                local_params[b, t] = proposals[b, t - 1]
            else:
                local_params[b, t] = local_params[b, t - 1]

        local_params[b, :] = scaled_sigmoid(local_params[b, :], lower, upper)

    return local_params


@njit
def _one_step_jump(
    x: np.ndarray,
    p_jump: np.ndarray,
    proposal: np.ndarray,
) -> np.ndarray:
    # Sample one step of jump process.
    jump = np.random.binomial(1, p_jump, size=x.shape[0])
    x_next = np.where(jump, proposal, x)
    return x_next


class Jump(Transition):
    """
    Simple jump transition.

    At each step the value either stays the same or jumps to a new sample drawn
    from a proposal prior.
    """

    def __init__(
        self,
        bounds: Tuple[float, float],
        initial_prior=None,
        p_jump: float | Prior = 0.1,
        proposal_prior: Prior | None = None,
    ):
        """
        Initialize the jump transition.

        Parameters
        ----------
        bounds : tuple of float
            Parameter bounds (lower, upper).
        initial_prior : Prior, optional
            Prior distribution for the initial state.
        p_jump : float or Prior, optional
            Probability of jumping at each step. Default is 0.1.
        proposal_prior : Prior, optional
            Prior distribution used for jump proposals. Default is standard normal.
        """
        super().__init__(bounds, initial_prior)

        self.hyper_specs = {
            "p_jump": p_jump,
        }
        self.proposal_prior = proposal_prior or Prior("normal", loc=0.0, scale=1.0)
        self.transition_type = "jump"

    def _expand_to_batch(self, x, batch_size: int):
        # Expand scalar values to batch-sized arrays.
        if np.ndim(x) == 0:
            return np.full(batch_size, x, dtype=self.dtype)
        return x

    def sample(self, batch_size: int, steps: int) -> Dict[str, Any]:
        """
        Generate jump process trajectories.

        Parameters
        ----------
        batch_size : int
            Number of independent trajectories.
        steps : int
            Number of time steps per trajectory.

        Returns
        -------
        dict
            Dictionary containing:
            - 'local_params': np.ndarray of shape (batch_size, steps)
            - 'hyper_params': dict of sampled hyperparameters
            - 'fixed_params': dict of fixed hyperparameters
        """
        local_params = np.empty((batch_size, steps), dtype=self.dtype)
        local_params[:, 0] = self.initial_prior.sample(batch_size)

        hyper_params, fixed_params = self._resolve_hyperparams(batch_size)

        p_jump = self._expand_to_batch(
            hyper_params.get("p_jump", fixed_params["p_jump"]),
            batch_size,
        )

        proposals = self.proposal_prior.sample(batch_size * (steps - 1)).reshape(
            batch_size,
            steps - 1,
        )

        local_params = _sample_jump_process(
            local_params,
            p_jump,
            proposals,
            self.bounds,
        )

        return {
            "local_params": local_params,
            "hyper_params": hyper_params,
            "fixed_params": fixed_params,
        }

    def one_step(self, x: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """
        Sample one step from the jump transition.

        Parameters
        ----------
        x : np.ndarray
            Current state, shape (batch,)
        params : dict
            Resolved parameters containing 'p_jump'

        Returns
        -------
        np.ndarray
            Next state, shape (batch,)
        """
        p_jump = self._expand_to_batch(params['p_jump'], x.shape[0])
        proposal = self.proposal_prior.sample(x.shape[0])
        return _one_step_jump(x, p_jump, proposal)
