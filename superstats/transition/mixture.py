from typing import Sequence, Tuple, Dict, Any
import numpy as np
import warnings

from .transition import Transition, Prior
from superstats.utils.transformations import scaled_sigmoid

class Mixture(Transition):

    def __init__(
        self,
        transitions: Sequence[Transition],
        mixture_weights: Prior | Tuple[float, ...] | None = None,
        bounds: Tuple[float, float] | None = None,
        initial_prior: Prior | None = None,
        names: Sequence[str] | None = None,
    ):
        super().__init__(
            bounds=bounds,
            initial_prior=initial_prior,
        )

        self.transitions = list(transitions)

        if len(self.transitions) < 2:
            raise ValueError(
                "Mixture must contain at least two transitions."
            )

        self.K = len(self.transitions)

        # enforce shared latent state-space semantics
        for i, t in enumerate(self.transitions):
            if t._user_defined_bounds:
                raise ValueError(
                    f"Transition at index {i} "
                    f"({t.transition_type}) defines bounds={t.bounds}. "
                    "Transitions inside Mixture cannot define bounds. "
                    "Specify bounds in Mixture(...) instead."
                )
            if t._user_defined_initial_prior:
                raise ValueError(
                    f"Transition at index {i} "
                    f"({t.transition_type}) defines initial_prior. "
                    "Transitions inside Mixture cannot define "
                    "initial_prior. "
                    "Specify initial_prior in Mixture(...) instead."
                )
            if t.transition_type == "jump":
                if getattr(t, "_user_defined_p_jump", False):
                    raise ValueError(
                        f"Jump transition at index {i} defines p_jump. "
                        "Inside Mixture, Jump() must use p_jump=1. "
                        "Mixture weights already define jump probability."
                    )

            t.bounds = self.bounds
            t.initial_prior = self.initial_prior
            

        self.names = names or [t.transition_type for t in self.transitions]

        if len(self.names) != self.K:
            raise ValueError(
                "names must match number of transitions"
            )

        if isinstance(mixture_weights, (int, float)):
            raise TypeError(
                "mixture_weights must be tuple/list, "
                "Dirichlet Prior, or None. "
                "Scalar values are ambiguous."
            )

        if isinstance(mixture_weights, tuple):
            mixture_weights = list(mixture_weights)

        if isinstance(mixture_weights, list):
            w = np.asarray(
                mixture_weights,
                dtype=self.dtype
            )

            if w.shape[0] != self.K:
                raise ValueError(
                    f"mixture_weights length "
                    f"{w.shape[0]} != number "
                    f"of transitions {self.K}"
                )

            if np.any(w < 0):
                raise ValueError(
                    "mixture_weights contains "
                    "negative values"
                )

            s = w.sum()

            if not np.isclose(s, 1.0):
                warnings.warn(
                    f"mixture_weights sum to "
                    f"{s:.2f}, normalizing "
                    f"to simplex.",
                    RuntimeWarning
                )
                w = w / s

            self.mixture_weights = tuple(w.tolist())

        elif isinstance(mixture_weights, Prior):
            if mixture_weights.dist != "dirichlet":
                raise ValueError(
                    "mixture_weights Prior must "
                    "be 'dirichlet' to define "
                    "simplex-distributed weights"
                )
            self.mixture_weights = mixture_weights
        elif mixture_weights is None:
            self.mixture_weights = None
        else:
            raise TypeError(
                "Invalid type for mixture_weights"
            )

        self.transition_type = "mixture"

    def _sample_mixture_weights(
        self,
        batch_size: int
    ) -> np.ndarray:

        # Dirichlet prior
        if isinstance(self.mixture_weights, Prior):
            w = (
                self.mixture_weights
                .sample(batch_size)
                .astype(self.dtype)
            )

            return w

        # fixed weights
        if isinstance(self.mixture_weights, tuple):
            w = np.asarray(
                self.mixture_weights,
                dtype=self.dtype
            )

            return np.tile(w, (batch_size, 1))

        # default uniform weights
        w = np.ones(
            self.K,
            dtype=self.dtype
        ) / self.K

        return np.tile(w, (batch_size, 1))

    def _sample_regimes(
        self,
        weights: np.ndarray,
        num_steps: int
    ) -> np.ndarray:

        batch_size = weights.shape[0]

        regimes = np.zeros(
            (batch_size, num_steps),
            dtype=np.int32
        )

        for b in range(batch_size):
            regimes[b] = np.random.choice(
                self.K,
                size=num_steps,
                p=weights[b]
            )

        return regimes

    def sample(
        self,
        batch_size: int,
        num_steps: int
    ) -> Dict[str, Any]:

        if self.initial_prior is None:

            raise ValueError(
                "Mixture requires initial_prior. "
                "Specify initial_prior in "
                "Mixture(...)."
            )

        local_params = np.empty(
            (batch_size, num_steps),
            dtype=self.dtype
        )

        # initial state
        local_params[:, 0] = (
            self.initial_prior
            .sample(batch_size)
            .astype(self.dtype)
        )

        # mixture weights + regimes
        weights = self._sample_mixture_weights(
            batch_size
        )

        regimes = self._sample_regimes(
            weights,
            num_steps
        )

        # resolve transition hyperparameters
        resolved_params = []

        for model in self.transitions:
            hyper, fixed = model._resolve_hyperparams(
                batch_size
            )

            params: Dict[str, np.ndarray] = {}

            # sampled hyperparameters
            for k, v in hyper.items():
                params[k] = v.astype(self.dtype)

            # fixed hyperparameters
            for k, v in fixed.items():
                params[k] = np.full(
                    batch_size,
                    v,
                    dtype=self.dtype
                )

            resolved_params.append({
                "params": params,
                "hyper": hyper,
                "fixed": fixed,
            })

        for b in range(batch_size):          
            for t in range(1, num_steps):

                k = regimes[b, t]
                model = self.transitions[k]
                params_all = resolved_params[k]["params"]

                params = {
                    key: float(val[b]) if hasattr(val, "__len__") else float(val)
                    for key, val in params_all.items()
                }

                local_params[b, t] = model.sample_one_step(
                    local_params[b, t - 1],
                    params,
                )

            local_params[b, :] = scaled_sigmoid(
                local_params[b, :], 
                self.bounds[0], 
                self.bounds[1]
            )

        # collect outputs
        hyper_params: Dict[str, np.ndarray] = {}
        fixed_params: Dict[str, float] = {}

        for name, resolved in zip(
            self.names,
            resolved_params
        ):

            # sampled hyperparameters
            for k, v in resolved["hyper"].items():
                hyper_params[f"{name}_{k}"] = v.astype(self.dtype)

            # fixed hyperparameters
            for k, v in resolved["fixed"].items():
                fixed_params[f"{name}_{k}"] = float(v)

        if isinstance(self.mixture_weights, Prior):
            hyper_params["mixture_weights"] = (
                weights
            )
        else:
            fixed_params["mixture_weights"] = (
                self.mixture_weights
                if self.mixture_weights is not None
                else tuple(
                    np.ones(self.K) / self.K
                )
            )

        return {
            "local_params": local_params,
            "regimes": regimes,
            "hyper_params": hyper_params,
            "fixed_params": fixed_params,
        }