from __future__ import annotations

from typing import Sequence, Tuple, Dict, Any
import numpy as np

from .transition import Transition, Prior


class Mixture(Transition):

    def __init__(
        self,
        transitions: Sequence[Transition],
        mixture_prob: Prior | Tuple[float, ...] | None = None,
        bounds: Tuple[float, float] | None = None,
        names: Sequence[str] | None = None,
    ):
        super().__init__(bounds=bounds)

        self.transitions = list(transitions)

        if len(self.transitions) < 2:
            raise ValueError("Mixture must contain at least two transitions.")

        self.K = len(self.transitions)

        self.names = names or [t.transition_type + f"_{i}" for i, t in enumerate(self.transitions)]

        if len(self.names) != self.K:
            raise ValueError("names must match number of transitions")

        self.mixture_prob = mixture_prob
        self.transition_type = "mixture"

    # -------------------------
    # mixture weights
    # -------------------------

    def _sample_mixture_weights(self, batch_size: int) -> np.ndarray:

        if isinstance(self.mixture_prob, Prior):
            w = self.mixture_prob.sample(batch_size).astype(self.dtype)
            return w / w.sum(axis=1, keepdims=True)

        if isinstance(self.mixture_prob, tuple):
            w = np.asarray(self.mixture_prob, dtype=self.dtype)
            w = w / w.sum()
            return np.tile(w, (batch_size, 1))

        w = np.ones(self.K, dtype=self.dtype) / self.K
        return np.tile(w, (batch_size, 1))

    def _sample_regimes(self, weights: np.ndarray, steps: int) -> np.ndarray:

        batch_size = weights.shape[0]
        regimes = np.zeros((batch_size, steps), dtype=np.int32)

        for b in range(batch_size):
            regimes[b] = np.random.choice(self.K, size=steps, p=weights[b])

        return regimes

    # -------------------------
    # parameter flattening
    # -------------------------

    def _collect_model_params(self, batch_size: int) -> tuple[Dict[str, np.ndarray], Dict[str, float]]:

        hyper: Dict[str, np.ndarray] = {}
        fixed: Dict[str, float] = {}

        for name, model in zip(self.names, self.transitions):

            h, f = model._resolve_hyperparams(batch_size)

            # hyper params
            for k, v in h.items():
                hyper[f"{name}.{k}"] = v

            # fixed params
            for k, v in f.items():
                fixed[f"{name}.{k}"] = v

        return hyper, fixed

    # -------------------------
    # sampling
    # -------------------------

    def sample(self, batch_size: int, steps: int) -> Dict[str, Any]:

        x = np.empty((batch_size, steps), dtype=self.dtype)

        x[:, 0] = self.transitions[0].initial_prior.sample(batch_size).astype(self.dtype)

        weights = self._sample_mixture_weights(batch_size)
        regimes = self._sample_regimes(weights, steps)

        for t in range(1, steps):
            for b in range(batch_size):
                k = regimes[b, t]

                x[b, t] = self.transitions[k].sample_one_step(
                    np.array([x[b, t - 1]]),
                    {},
                )[0]

        hyper_params, fixed_params = self._collect_model_params(batch_size)

        # mixture weights are hyperparams
        hyper_params["mixture_weights"] = weights

        return {
            "local_params": x,
            "regimes": regimes,
            "hyper_params": hyper_params,
            "fixed_params": fixed_params,
        }