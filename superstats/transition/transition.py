from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

from superstats.prior import Prior


class Transition(ABC):
    """
    Base class for parameter transition processes.

    A transition encapsulates any stochastic process used to evolve a
    parameter through time.  Concrete subclasses implement :meth:`sample`
    which returns a batch of trajectories.
    """

    def __init__(
        self,
        bounds: Tuple[float, float] | np.ndarray,
        initial_prior: None | Prior = None
    ):
        """
        Parameters
        ----------
        bounds : tuple or np.ndarray
            Lower and upper bounds for the parameter values.
        initial_prior : Prior, optional
            Prior distribution for the initial step; defaults to a standard
            normal.
        """
        if initial_prior is None:
            self.initial_prior = Prior(
                "normal", loc=0.0, scale=1.0
            )
        else:
            self.initial_prior = initial_prior

        self.bounds = np.asarray(bounds, dtype=np.float32)

    @abstractmethod
    def sample(self, batch_size: int, steps: int) -> np.ndarray:
        """Generate a batch of trajectories for this transition.

        Parameters
        ----------
        batch_size : int
            Number of independent trajectories to sample.
        steps : int
            Length of each trajectory (number of time points).

        Returns
        -------
        np.ndarray
            Samples for the requested trajectories; shape is implementation-dependent.
        """
        ...