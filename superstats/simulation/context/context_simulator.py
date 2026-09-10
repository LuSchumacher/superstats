"""Wrapper for context simulation."""

from typing import Callable, Dict
import numpy as np


class ContextSimulator:
    """Wrapper for context simulation.

    Parameters
    ----------
    simulator: Callable
        The simulator function to use.
    is_batched: bool, default: True
        Whether ``simulator`` accepts a ``batch_size`` argument and returns
        batched outputs. If ``False``, the simulator is called once per
        requested batch element and its outputs are stacked along the first axis.
    """

    def __init__(self, simulator: Callable, is_batched: bool = True):
        self.simulator = simulator
        self.is_batched = is_batched

    def sample(self, batch_size: int, num_steps: int) -> Dict[str, np.ndarray]:
        """Sample context.

        Parameters
        ----------
        batch_size : int
            Number of independent simulation batches to generate.
        num_steps  : int
            Number of time steps per trajectory.

        Returns
        -------
            Dict[str, np.ndarray]: The simulated data.
        """
        if self.is_batched:
            return self.simulator(
                batch_size=batch_size,
                num_steps=num_steps,
            )

        simulations = [self.simulator(num_steps=num_steps) for _ in range(batch_size)]
        if not simulations:
            return {}

        keys = list(simulations[0])
        if any(list(simulation) != keys for simulation in simulations[1:]):
            raise ValueError("Non-batched context simulator output keys must be consistent.")

        return {key: np.stack([np.asarray(simulation[key]) for simulation in simulations], axis=0) for key in keys}
