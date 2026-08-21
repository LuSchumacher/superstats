"""Wrapper for a contaminated random choice data augmentation process."""

from numbers import Real

import numpy as np

from .contamination import ContaminationProcess
from superstats.defaults import DEFAULT_P_CONTAMINATED_PRIOR
from superstats.prior.prior import Prior
from superstats.transition import DeterministicTransition, StochasticTransition


class RandomChoiceContamination(ContaminationProcess):
    """Contamination at random for diffusion models with a per-dataset
    contamination probability.

    Contamination is drawn per (batch, step): whenever a time step is
    selected as contaminated, both the response time and the choice at
    that step are replaced by draws from a contaminant distribution. Any
    other keys present in `data` are passed through unchanged.
    Non-positive response times are treated as non-finished trials: they are
    left unchanged and excluded from the contaminant distributions.

    Contaminant response times are drawn from a heavy-tailed (Student's t)
    distribution centered on each dataset's own log-RT mean and scaled by
    its own log-RT standard deviation, so contaminants stay plausible in
    scale for that dataset while still being outliers relative to it [1].

    Contaminant choices are drawn either from the observed unique choice
    values (if choices are discrete) or from a uniform distribution over
    the observed choice range (if choices are continuous); this is
    determined once from the whole batch, not per dataset.

    [1] Wu, Y., Radev, S. T., & Tuerlinckx, F. (2026). Testing and improving the
        robustness of amortized Bayesian inference for cognitive models.
        Psychological Methods. https://arxiv.org/abs/2412.20586

    Parameters
    ----------
    p_contaminated : float, Prior, StochasticTransition,
        DeterministicTransition, or None, default: None
        Probability that a time step is contaminated.
        - None (default): drawn from `DEFAULT_P_CONTAMINATED_PRIOR`.
        - float: fixed probability, shared across the whole batch.
        - Prior: sampled once per dataset to obtain a per-dataset
          probability.
        - StochasticTransition: sampled once per dataset and time step.
        - DeterministicTransition: constructs one probability trajectory
          per dataset.
    infer : bool, default: False
        Whether `Model` should register contamination parameters as model
        parameters. The parameter type determines its category in the same
        way as `JointPrior`: scalars are fixed, `Prior` values are shared,
        stochastic trajectories are local, and deterministic trajectories
        are deterministic. Transition hyperparameters are registered as
        hyper or fixed parameters as appropriate. A scalar remains fixed
        even when `infer=True`, because it has no uncertainty to infer.
    student_t_df : float, default: 5
        Degrees of freedom for the Student's t distribution used to
        generate contaminant response times. Must be greater than 2.
    response_time_key : str, default: "response_time"
        Key in `data` containing response times.
    choice_key : str, default: "choice"
        Key in `data` containing choices.
    """

    def __init__(
        self,
        p_contaminated: float | Prior | StochasticTransition | DeterministicTransition | None = None,
        infer: bool = False,
        student_t_df: float = 5,
        response_time_key: str = "response_time",
        choice_key: str = "choice",
    ):
        if student_t_df <= 2:
            raise ValueError("student_t_df must be greater than 2.")
        if response_time_key == choice_key:
            raise ValueError("response_time_key and choice_key must be different.")
        if not isinstance(infer, bool):
            raise TypeError("infer must be a bool.")

        self.p_contaminated = p_contaminated if p_contaminated is not None else DEFAULT_P_CONTAMINATED_PRIOR
        if not (
            isinstance(self.p_contaminated, Real)
            or isinstance(self.p_contaminated, (Prior, StochasticTransition, DeterministicTransition))
        ):
            raise TypeError(
                "p_contaminated must be a scalar, Prior, StochasticTransition, DeterministicTransition, or None."
            )
        if isinstance(self.p_contaminated, Real) and not 0.0 <= float(self.p_contaminated) <= 1.0:
            raise ValueError("p_contaminated must be between 0 and 1.")
        if isinstance(self.p_contaminated, (StochasticTransition, DeterministicTransition)):
            lower, upper = self.p_contaminated.bounds
            if lower < 0.0 or upper > 1.0:
                raise ValueError("A p_contaminated transition must have bounds within [0, 1].")

        self.infer = infer
        self.student_t_df = student_t_df
        self.key_map = {
            "response_time": response_time_key,
            "choice": choice_key,
        }
        self.required_keys = set(self.key_map.values())

    def _draw_parameter_groups(self, batch_size: int, num_steps: int) -> dict[str, dict]:
        """Draw contamination probabilities and group them like `JointPrior`.

        Note: `Prior.sample` draws from the global `np.random` state, not
        from the `rng` passed into `apply`, so draws from a `Prior` are not
        controlled by the seed threaded through `Model.sample`.
        """
        p = self.p_contaminated

        if isinstance(p, Prior):
            groups = {"shared_params": {"p_contaminated": p.sample(batch_size)}}
        elif isinstance(p, StochasticTransition):
            samples = p.sample(batch_size=batch_size, num_steps=num_steps)
            groups = {
                "local_params": {"p_contaminated": samples["local_params"]},
                "hyper_params": {f"p_contaminated_{key}": value for key, value in samples["hyper_params"].items()},
                "fixed_params": {f"p_contaminated_{key}": value for key, value in samples["fixed_params"].items()},
            }
        elif isinstance(p, DeterministicTransition):
            samples = p.sample(batch_size=batch_size, num_steps=num_steps)
            groups = {
                "deterministic_params": {"p_contaminated": samples["deterministic_params"]},
                "hyper_params": {f"p_contaminated_{key}": value for key, value in samples["hyper_params"].items()},
                "fixed_params": {f"p_contaminated_{key}": value for key, value in samples["fixed_params"].items()},
            }
        else:
            groups = {"fixed_params": {"p_contaminated": p}}

        values = next(values["p_contaminated"] for values in groups.values() if "p_contaminated" in values)
        if np.any((np.asarray(values) < 0.0) | (np.asarray(values) > 1.0)):
            raise ValueError("Sampled p_contaminated values must be between 0 and 1.")

        return groups

    def parameter_groups(self) -> dict[str, list[str]]:
        """Return parameter names grouped for registration on `Model`."""
        if not self.infer:
            return {}

        groups = self._draw_parameter_groups(batch_size=1, num_steps=1)
        return {group: list(values) for group, values in groups.items() if values}

    @staticmethod
    def _choice_is_discrete(choice: np.ndarray) -> bool:
        """Return whether numeric choices are integer-valued."""
        if np.issubdtype(choice.dtype, np.integer) or np.issubdtype(choice.dtype, np.bool_):
            return True
        return bool(np.all(choice == np.floor(choice)))

    @staticmethod
    def _log_rt_stats(response_time: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute per-dataset log-RT mean/std over valid, positive RTs."""
        stats_mask = mask & (response_time > 0)
        counts = stats_mask.sum(axis=1, keepdims=True)

        log_rt = np.zeros(response_time.shape)
        np.log(response_time, out=log_rt, where=stats_mask)

        sums = np.sum(log_rt, axis=1, keepdims=True)
        mean_rt = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)

        centered = np.where(stats_mask, log_rt - mean_rt, 0.0)
        squared_sums = np.sum(centered**2, axis=1, keepdims=True)
        var_rt = np.divide(squared_sums, counts, out=np.zeros_like(squared_sums), where=counts > 0)
        std_rt = np.sqrt(var_rt)

        return mean_rt, std_rt

    def apply(
        self,
        data: dict[str, np.ndarray],
        rng: np.random.Generator | None = None,
    ) -> dict:
        """Apply random-choice contamination to response times and choices.

        Parameters
        ----------
        data          : dict with at least the configured response-time and
            choice keys, each an np.ndarray of shape (batch_size, num_steps).
            Any additional keys are passed through unchanged.
        rng           : np.random.Generator or None, optional, default: None
            Random generator to use. If None, a fresh, unseeded generator
            is created via `_default_rng`, so calling `apply` directly is
            safe but not reproducible unless a seeded `rng` is supplied.

        Returns
        -------
        result : dict
            A shallow copy of `data` with the configured response-time and
            choice keys replaced by their contaminated versions, plus
            "p_contaminated" (the contamination probability used, shape
            (batch_size,) for a scalar or `Prior`, and
            (batch_size, num_steps) for a transition). With `infer=True`,
            inferred transition hyperparameters and fixed transition
            parameters are also included using `p_contaminated_`-prefixed
            names. All other keys in `data` are carried over unchanged.

        Raises
        ------
        KeyError
            If `data` is missing either configured required key.
        """
        rng = self._default_rng(rng)

        missing = self.required_keys - data.keys()
        if missing:
            raise KeyError(f"data is missing required key(s): {sorted(missing)}")

        response_time_key = self.key_map["response_time"]
        choice_key = self.key_map["choice"]
        response_time = data[response_time_key]
        choice = data[choice_key]

        batch_size, num_steps = response_time.shape
        parameter_groups = self._draw_parameter_groups(batch_size, num_steps)
        p = next(values["p_contaminated"] for values in parameter_groups.values() if "p_contaminated" in values)
        p = np.asarray(p)
        if p.ndim == 0:
            p = np.full(batch_size, p.item())

        valid_rt = response_time > 0
        probabilities = p[:, None] if p.ndim == 1 else p
        mask = (rng.random((batch_size, num_steps)) < probabilities) & valid_rt
        n_contaminated = mask.sum()

        # contaminant response times
        mean_rt, std_rt = self._log_rt_stats(response_time, valid_rt)

        student_samples = rng.standard_t(df=self.student_t_df, size=(batch_size, num_steps))
        contaminant_log_rt = mean_rt + student_samples * std_rt * np.sqrt((self.student_t_df - 2) / self.student_t_df)
        contaminant_rt = np.exp(contaminant_log_rt)

        # contaminant choices
        contaminated_choices = choice.copy()
        if n_contaminated > 0:
            valid_choices = choice[valid_rt]
            is_discrete = self._choice_is_discrete(valid_choices)

            if is_discrete:
                unique_choices = np.unique(valid_choices)
                contaminant_choices = rng.choice(unique_choices, size=n_contaminated)
            else:
                contaminant_choices = rng.uniform(valid_choices.min(), valid_choices.max(), size=n_contaminated)

            contaminated_choices[mask] = contaminant_choices

        out = dict(data)
        out[response_time_key] = np.where(mask, contaminant_rt, response_time)
        out[choice_key] = contaminated_choices
        out["p_contaminated"] = p

        if self.infer:
            for values in parameter_groups.values():
                out.update(values)

        return out
