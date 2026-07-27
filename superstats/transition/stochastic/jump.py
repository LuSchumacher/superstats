"""Jump-process transition models."""

from typing import Tuple, Dict, Any
import numpy as np
from numba import njit, prange

from .stochastic_transition import StochasticTransition, Prior
from superstats.utils.transformations import scaled_sigmoid


@njit(parallel=True, fastmath=True)
def _sample_jump_process(
    local_params: np.ndarray,
    p_jump: np.ndarray,
    proposals: np.ndarray,
    bounds: np.ndarray,
) -> np.ndarray:
    """Vectorized jump-process rollout across a batch, filled in place.

    Parameters
    ----------
    local_params : np.ndarray of shape (batch_size, steps)
        Pre-allocated trajectory array; `local_params[:, 0]` must already
        hold the initial state. Overwritten in place with the full rollout.
    p_jump       : np.ndarray of shape (batch_size,)
        Per-trajectory probability of jumping at each step.
    proposals    : np.ndarray of shape (batch_size, steps - 1)
        Pre-sampled proposal values to jump to, one per step.
    bounds       : np.ndarray of shape (2,)
        (lower, upper) bounds passed to `scaled_sigmoid`.

    Returns
    -------
    local_params : np.ndarray of shape (batch_size, steps) - the same
        array, filled with the bounded jump-process rollout
    """
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


@njit
def _one_step_jump(
    x: float,
    p_jump: float,
    proposal: float,
) -> float:
    """Advance a single jump-process state by one step.

    Parameters
    ----------
    x        : float
        Previous latent state.
    p_jump   : float
        Probability of jumping at this step.
    proposal : float
        Candidate value to jump to if a jump occurs.

    Returns
    -------
    x_next : float - the next latent state (either `x` or `proposal`)
    """
    if np.random.rand() < p_jump:
        return proposal
    return x


class Jump(StochasticTransition):
    """Simple jump process: stay or jump to a proposal draw.

    Parameters
    ----------
    bounds         : tuple or None, optional, default: None
        Lower and upper bounds for the latent state.
    initial_prior  : Prior or None, optional, default: None
        Prior for the initial latent state.
    p_jump         : float or Prior, optional, default: 1.0
        Probability of jumping at each step (or a Prior to infer per-batch).
    proposal_prior : Prior or None, optional, default: None
        Prior from which to draw proposal values when a jump occurs. Falls
        back to a standard normal `Prior` if not provided.

    Notes
    -----
    At each step the process either stays at the previous value or jumps
    to an independent proposal sampled from `proposal_prior`.
    """

    def __init__(
        self,
        bounds: Tuple[float, float] | None = None,
        initial_prior: Prior | None = None,
        p_jump: float | Prior = 1.0,
        proposal_prior: Prior | None = None,
    ):
        super().__init__(bounds, initial_prior)

        self._user_defined_p_jump = p_jump != 1.0

        self.hyper_specs = {
            "p_jump": p_jump,
        }

        self.proposal_prior = proposal_prior or Prior("normal", loc=0.0, scale=1.0)
        self.transition_name = "jump"

    def sample(self, batch_size: int, num_steps: int) -> Dict[str, Any]:
        """Draw `batch_size` jump-process trajectories of length `num_steps`.

        Parameters
        ----------
        batch_size : int
            Number of independent trajectories to draw.
        num_steps  : int
            Number of time steps per trajectory.

        Returns
        -------
        result : dict - dictionary with keys `local_params`,
            `hyper_params`, and `fixed_params`
        """
        local_params = np.empty((batch_size, num_steps), dtype=self.dtype)
        local_params[:, 0] = self.initial_prior.sample(batch_size)

        hyper, fixed = self._resolve_hyperparams(batch_size)

        if "p_jump" in hyper:
            p_jump = hyper["p_jump"]
        else:
            p_jump = np.full(batch_size, fixed["p_jump"], dtype=self.dtype)

        proposals = self.proposal_prior.sample(batch_size * (num_steps - 1)).reshape(
            batch_size,
            num_steps - 1,
        )

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

    def sample_one_step(self, x: float, params: Dict[str, Any]) -> float:
        """Take one step of the jump process.

        Parameters
        ----------
        x      : float
            Previous latent state.
        params : dict
            Expected key: `p_jump`.

        Returns
        -------
        x_next : float - the next latent state
        """
        return _one_step_jump(
            x,
            params["p_jump"],
            self.proposal_prior.sample(1)[0],
        )
