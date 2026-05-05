from typing import Tuple, Dict, Any
from collections.abc import Sequence
import numpy as np
from numba import njit, prange

from .transition import Transition, Prior
from superstats.utils.transformations import scaled_sigmoid


class Mixture(Transition):

    def __init__(
        self,
        transitions: Sequence[Transition],
        mixture_prob: Prior | Tuple | None = None,
        bounds: Tuple[float, float] | None = None,
        initial_prior: Prior | None = None,
        dtype: np.dtype = np.float32,
    ):

        self.transitions = transitions

        if len(self.transitions) < 2:
            raise ValueError("Mixture requires at least 2 transitions")

        super().__init__(bounds, initial_prior, dtype)

        if mixture_prob is None:
            self.mixture_prob = mixture_prob
            self.n_transitions = len(self.transitions)
            self.hyper_specs = {"mixture_prob": mixture_prob}

    def sample(self, batch_size: int, steps: int) -> Dict[str, Any]:
        """
        Generate mixture transition trajectories.

        At each time step, randomly selects one transition based on mixture
        probabilities and uses its output.

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
            - 'hyper_params': dict with sampled mixture probabilities and all transition params
            - 'fixed_params': dict with fixed parameters from transitions
        """
        local_params = np.empty((batch_size, steps), dtype=self.dtype)
        local_params[:, 0] = self.initial_prior.sample(batch_size)

        # Sample mixture probabilities
        mixture_probs = self.mixture_prob.sample(batch_size)
        
        # Handle case where mixture_prob returns 1D array (single value per batch)
        if mixture_probs.ndim == 1:
            # For Beta: shape is (batch_size,), expand to (batch_size, 2)
            probs = np.column_stack([mixture_probs, 1.0 - mixture_probs])
        else:
            # For Dirichlet or multi-dimensional: shape is (batch_size, n_transitions)
            probs = mixture_probs

        # Sample trajectories and parameters for all transitions
        all_trajectories = []
        all_hyper_params = {}
        fixed_params_all = {}
        
        for i, transition in enumerate(self.transitions):
            result = transition.sample(batch_size, steps)
            all_trajectories.append(result["local_params"])
            # Store hyperparams with transition index prefix
            for key, val in result["hyper_params"].items():
                all_hyper_params[f"{self.transition_names[i]}_{key}"] = val
            # Merge fixed params from all transitions
            fixed_params_all.update(result["fixed_params"])

        # Build trajectory by selecting from transitions at each step
        for t in range(1, steps):
            # Sample which transition to use for each batch
            transitions_idx = np.array([
                np.random.choice(self.n_transitions, p=probs[b])
                for b in range(batch_size)
            ])

            # Select values from chosen transitions
            for b in range(batch_size):
                local_params[b, t] = all_trajectories[transitions_idx[b]][b, t]

        hyper_params = {
            "mixture_prob": mixture_probs,
            **all_hyper_params
        }

        return {
            "local_params": local_params,
            "hyper_params": hyper_params,
            "fixed_params": fixed_params_all,
        }

    def one_step(self, x: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """
        Sample one step from the mixture transition.

        Randomly selects one transition and applies its one_step method.

        Parameters
        ----------
        x : np.ndarray
            Current state, shape (batch,)
        params : dict
            Resolved parameters containing 'mixture_prob' and transition hyperparameters

        Returns
        -------
        np.ndarray
            Next state, shape (batch,)
        """
        batch_size = x.shape[0]
        mixture_probs = params["mixture_prob"]

        # Handle 1D mixture probs (from Beta)
        if mixture_probs.ndim == 1:
            probs = np.column_stack([mixture_probs, 1.0 - mixture_probs])
        else:
            probs = mixture_probs

        x_next = np.empty(batch_size, dtype=self.dtype)

        # For each batch element, select a transition and apply it
        for b in range(batch_size):
            transition_idx = np.random.choice(self.n_transitions, p=probs[b])
            selected_transition = self.transitions[transition_idx]
            transition_name = self.transition_names[transition_idx]

            # Extract hyperparameters for this transition from params
            transition_params = {}
            prefix = f"{transition_name}_"
            
            for key, val in params.items():
                if key.startswith(prefix):
                    # Remove prefix from key
                    param_name = key[len(prefix):]
                    # Extract batch element
                    if isinstance(val, np.ndarray) and val.shape[0] == batch_size:
                        transition_params[param_name] = val[b:b+1]
                    else:
                        transition_params[param_name] = val
            
            # Also include any fixed params that might be needed
            for key, val in params.items():
                if not key.startswith(f"{self.transition_names[0]}_") and \
                   not any(key.startswith(f"{tn}_") for tn in self.transition_names) and \
                   key != "mixture_prob":
                    transition_params[key] = val

            # Call selected transition's one_step
            x_step = selected_transition.one_step(x[b:b+1], transition_params)
            x_next[b] = x_step[0]

        return x_next

