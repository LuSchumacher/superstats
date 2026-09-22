"""Wrapper for a contaminated random choice data augmentation process."""

import numpy as np

from .contamination import ContaminationProcess


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
    infer : bool, default: False
        Include the probability and its sampled transition hyperparameters in
        inference targets. Specify ``p_contaminated`` in ``JointPrior``;
        otherwise Model supplies the default beta prior. With False, it is
        a nuisance parameter drawn afresh during posterior resimulation.
    student_t_df      : float, default: 5
        Degrees of freedom for the Student's t distribution used to
        generate contaminant response times. Must be greater than 2.
    response_time_key : str, default: "response_time"
        Key in `data` containing response times.
    choice_key : str, default: "choice"
        Key in `data` containing choices.
    """

    def __init__(
        self,
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

        self.infer = infer
        self.student_t_df = student_t_df
        self.key_map = {
            "response_time": response_time_key,
            "choice": choice_key,
        }
        self.required_keys = set(self.key_map.values())

    def __call__(self, data, rng=None, *, probability):
        return self.apply(data, rng=rng, probability=probability)

    def apply(
        self,
        data: dict[str, np.ndarray],
        rng: np.random.Generator | None = None,
        *,
        probability: np.ndarray | float,
    ) -> dict:
        """Apply random-choice contamination to response times and choices.

        Parameters
        ----------
        data : dict with at least the configured response-time and
            choice keys, each an np.ndarray of shape (batch_size, num_steps).
            Any additional keys are passed through unchanged.
        rng  : np.random.Generator or None, optional, default: None
            Random generator to use. If None, a fresh, unseeded generator
            is created via `_default_rng`, so calling `apply` directly is
            safe but not reproducible unless a seeded `rng` is supplied.

        probability : float or np.ndarray
            Final linked probability supplied by Model, in [0, 1].

        Returns
        -------
        result : dict
            A shallow copy of `data` with the configured response-time and
            choice keys replaced by their contaminated versions, plus
            "p_contaminated" (the contamination probability used, shape
            (batch_size,) for a shared value, and
            (batch_size, num_steps) for a trajectory). The
            raw inference targets are returned by Model separately. All other
            keys in `data` are carried over unchanged.

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
        p = np.asarray(probability)
        if p.ndim == 3 and p.shape[-1] == 1:
            p = p[..., 0]
        if p.shape == (batch_size, 1):
            p = p[:, 0]
        if not np.all(np.isfinite(p)) or np.any((p < 0.0) | (p > 1.0)):
            raise ValueError(
                "Sampled p_contaminated values must be between 0 and 1; configure Model.latent_link_functions."
            )
        if p.ndim == 0:
            p = np.full(batch_size, p.item())

        if p.shape not in ((batch_size,), (batch_size, num_steps)):
            raise ValueError("probability must be scalar, per-dataset, or per-trial.")

        valid_rt = response_time > 0
        probabilities = p[:, None] if p.ndim == 1 else p
        mask = (rng.random((batch_size, num_steps)) < probabilities) & valid_rt
        n_contaminated = mask.sum()

        # contaminant response times
        counts = valid_rt.sum(axis=1, keepdims=True)
        log_rt = np.zeros(response_time.shape)
        np.log(response_time, out=log_rt, where=valid_rt)
        mean_rt = np.divide(
            log_rt.sum(axis=1, keepdims=True),
            counts,
            out=np.zeros_like(counts, dtype=float),
            where=counts > 0,
        )
        centered = np.where(valid_rt, log_rt - mean_rt, 0.0)
        std_rt = np.sqrt(
            np.divide(
                (centered**2).sum(axis=1, keepdims=True),
                counts,
                out=np.zeros_like(counts, dtype=float),
                where=counts > 0,
            )
        )

        student_samples = rng.standard_t(df=self.student_t_df, size=(batch_size, num_steps))
        contaminant_log_rt = mean_rt + student_samples * std_rt * np.sqrt((self.student_t_df - 2) / self.student_t_df)
        contaminant_rt = np.exp(contaminant_log_rt)

        # contaminant choices
        contaminated_choices = choice.copy()
        if n_contaminated > 0:
            valid_choices = choice[valid_rt]
            if (
                np.issubdtype(choice.dtype, np.integer)
                or np.issubdtype(choice.dtype, np.bool_)
                or np.all(valid_choices == np.floor(valid_choices))
            ):
                unique_choices = np.unique(valid_choices)
                contaminant_choices = rng.choice(unique_choices, size=n_contaminated)
            else:
                contaminant_choices = rng.uniform(valid_choices.min(), valid_choices.max(), size=n_contaminated)

            contaminated_choices[mask] = contaminant_choices

        out = dict(data)
        out[response_time_key] = np.where(mask, contaminant_rt, response_time)
        out[choice_key] = contaminated_choices
        out["p_contaminated"] = p

        return out
