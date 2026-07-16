"""Wrapper for a contaminated random choice data augmentation process."""

from .contamination_process import ContaminationProcess
from superstats.prior.prior import Prior

import numpy as np

DEFAULT_P_CONTAMINATED_PRIOR = Prior("beta", a=1.5, b=15)


class RandomChoiceProcess(ContaminationProcess):
    """Contamination at random, with a per-dataset contamination probability.

    Contamination is drawn per (batch, step): whenever a time step is
    selected as contaminated, both the response time and the choice at
    that step are replaced by draws from a contaminant distribution. Any
    other keys present in `data` are passed through unchanged.

    Contaminant response times are drawn from a heavy-tailed (Student's t)
    distribution centered on each dataset's own log-RT mean and scaled by
    its own log-RT standard deviation, so contaminants stay plausible in
    scale for that dataset while still being outliers relative to it.
    Contaminant choices are drawn either from the observed unique choice
    values (if choices are discrete) or from a uniform distribution over
    the observed choice range (if choices are continuous); this is
    determined once from the whole batch, not per dataset.

    Parameters
    ----------
    p_contaminated : float, Prior, or None, default: None
        Probability that a time step is contaminated.
        - None (default): drawn from `DEFAULT_P_CONTAMINATED_PRIOR`.
        - float: fixed probability, shared across the whole batch.
        - Prior: sampled once per dataset to obtain a per-dataset
          probability (i.e. `_draw_p` returns one value per batch
          element, not one shared value for the whole batch).
    """

    def __init__(
        self,
        p_contaminated: float | Prior | None = None,
    ):
        self.p_contaminated = p_contaminated if p_contaminated is not None else DEFAULT_P_CONTAMINATED_PRIOR

    def _draw_p(self, n: int) -> np.ndarray:
        """Return `n` contamination probabilities, one per dataset.

        Note: `Prior.sample` draws from the global `np.random` state, not
        from the `rng` passed into `apply`, so draws from a `Prior` are not
        controlled by the seed threaded through `GenerativeModel.sample`.
        """
        p = self.p_contaminated
        if isinstance(p, Prior):
            vals = p.sample(n)
        else:
            vals = np.full(n, p)
        return vals

    def apply(
        self,
        data: dict[str, np.ndarray],
        rng: np.random.Generator | None = None,
        student_t_df: int = 5,
    ) -> dict:
        """Apply random-choice contamination to response times and choices.

        Parameters
        ----------
        data          : dict with at least the keys "response_time" and
            "choice", each an np.ndarray of shape (batch_size, num_steps).
            Any additional keys are passed through unchanged.
        rng           : np.random.Generator or None, optional, default: None
            Random generator to use. If None, a fresh, unseeded generator
            is created via `_default_rng`, so calling `apply` directly is
            safe but not reproducible unless a seeded `rng` is supplied.
        student_t_df  : int, optional, default: 5
            Degrees of freedom for the Student's t distribution used to
            generate contaminant response times. Lower values produce
            heavier tails (more extreme contaminant RTs).

        Returns
        -------
        result : dict
            A copy of `data` with "response_time" and "choice" replaced by
            their contaminated versions, plus "p_contaminated" (the
            per-dataset contamination probability used, shape
            (batch_size,)). All other keys in `data` are carried over
            unchanged.

        Raises
        ------
        KeyError
            If `data` is missing the "response_time" or "choice" key.
        """
        rng = self._default_rng(rng)

        required_keys = {"response_time", "choice"}
        missing = required_keys - data.keys()
        if missing:
            raise KeyError(f"data is missing required key(s): {sorted(missing)}")

        response_time = data["response_time"]
        choice = data["choice"]

        batch_size, num_steps = response_time.shape
        p = self._draw_p(batch_size)
        mask = rng.random((batch_size, num_steps)) < p[:, None]
        n_contaminated = int(mask.sum())

        # contaminant response times
        log_rt = np.log(response_time)
        mean_rt = log_rt.mean(axis=1, keepdims=True)
        std_rt = log_rt.std(axis=1, keepdims=True)

        student_samples = rng.standard_t(df=student_t_df, size=(batch_size, num_steps))
        contaminant_log_rt = mean_rt + student_samples * std_rt * np.sqrt((student_t_df - 2) / student_t_df)
        contaminant_rt = np.exp(contaminant_log_rt)

        contaminated_rt = np.where(mask, contaminant_rt, response_time)

        # contaminant choices
        contaminated_choices = choice.copy()
        if n_contaminated > 0:
            unique_choices = np.unique(choice)
            is_discrete = np.all(np.equal(np.mod(unique_choices, 1), 0))

            if is_discrete:
                contaminant_choices = rng.choice(unique_choices, size=n_contaminated)
            else:
                contaminant_choices = rng.uniform(choice.min(), choice.max(), size=n_contaminated)

            contaminated_choices[mask] = contaminant_choices

        out = dict(data)
        out["response_time"] = contaminated_rt
        out["choice"] = contaminated_choices
        out["p_contaminated"] = p

        return out
