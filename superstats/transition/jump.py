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

@njit
def _one_step_jump(
    x: float,
    p_jump: float,
    proposal: float,
) -> float:
    if np.random.rand() < p_jump:
        return proposal
    return x

class Jump(Transition):
    """Simple jump process: stay or jump to a proposal draw.

    Parameters
    ----------
    bounds : tuple or None
        Lower and upper bounds for the latent state.
    initial_prior : Prior or None
        Prior for the initial latent state.
    p_jump : float or Prior
        Probability of jumping at each step (or a Prior to infer per-batch).
    proposal_prior : Prior or None
        Prior from which to draw proposal values when a jump occurs.

    Notes
    -----
    At each step the process either stays at the previous value or jumps
    to an independent proposal sampled from ``proposal_prior``.
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
        self.transition_type = "jump"

    def sample(self, batch_size: int, num_steps: int) -> Dict[str, Any]:
        """
        Draw `batch_size` jump-process trajectories of length `num_steps`.

        Parameters
        ----------
        batch_size : int
        num_steps : int

        Returns
        -------
        dict
            Dictionary with keys ``local_params``, ``hyper_params``, and
            ``fixed_params``.
        """

        local_params = np.empty((batch_size, num_steps), dtype=self.dtype)
        local_params[:, 0] = self.initial_prior.sample(batch_size).astype(self.dtype)

        hyper, fixed = self._resolve_hyperparams(batch_size)

        if "p_jump" in hyper:
            p_jump = hyper["p_jump"]
        else:
            p_jump = np.full(batch_size, fixed["p_jump"], dtype=self.dtype)

        proposals = self.proposal_prior.sample(batch_size * (num_steps - 1)).reshape(
            batch_size,
            num_steps - 1,
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

    def sample_one_step(self, x: float, params: Dict[str, Any]) -> float:
        """
        Take one step of the jump process.

        Parameters
        ----------
        x : float
            Previous latent state.
        params : dict
            Expect key ``p_jump``.

        Returns
        -------
        float
            Next latent state.
        """

        p_jump = float(params["p_jump"])
        proposal = float(self.proposal_prior.sample(1)[0])
        return _one_step_jump(
            x,
            p_jump,
            proposal,
        )