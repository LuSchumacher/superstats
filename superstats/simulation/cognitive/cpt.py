"""Cumulative-prospect-theory simulator."""

import numpy as np
from numba import njit, prange


@njit(fastmath=True, inline="always")
def _prelec(probability: float, gamma: float) -> float:
    if probability <= 0.0:
        return 0.0
    if probability >= 1.0:
        return 1.0
    return np.exp(-((-np.log(probability)) ** gamma))


@njit(fastmath=True)
def _cpt_utility(
    outcomes: np.ndarray,
    probabilities: np.ndarray,
    alpha: float,
    lamda: float,
    gamma: float,
) -> float:
    """Compute one option's CPT utility from outcomes and probabilities."""
    num_outcomes = outcomes.shape[0]
    values = np.empty(num_outcomes, dtype=np.float64)
    for k in range(num_outcomes):
        if outcomes[k] >= 0.0:
            values[k] = outcomes[k] ** alpha
        else:
            values[k] = -lamda * ((-outcomes[k]) ** alpha)

    losses = np.empty(num_outcomes, dtype=np.float64)
    loss_probabilities = np.empty(num_outcomes, dtype=np.float64)
    gains = np.empty(num_outcomes, dtype=np.float64)
    gain_probabilities = np.empty(num_outcomes, dtype=np.float64)
    num_losses = 0
    num_gains = 0
    for k in range(num_outcomes):
        if values[k] < 0.0 and probabilities[k] > 0.0:
            losses[num_losses] = values[k]
            loss_probabilities[num_losses] = probabilities[k]
            num_losses += 1
        elif values[k] >= 0.0 and probabilities[k] > 0.0:
            gains[num_gains] = values[k]
            gain_probabilities[num_gains] = probabilities[k]
            num_gains += 1

    utility = 0.0
    if num_losses > 0:
        loss_order = np.argsort(losses[:num_losses])
        cumulative = 0.0
        for rank in range(num_losses):
            index = loss_order[rank]
            previous = cumulative
            cumulative += loss_probabilities[index]
            weight = _prelec(cumulative, gamma) - _prelec(previous, gamma)
            utility += weight * losses[index]

    if num_gains > 0:
        gain_order = np.argsort(gains[:num_gains])
        cumulative = 0.0
        for rank in range(num_gains):
            index = gain_order[num_gains - rank - 1]
            previous = cumulative
            cumulative += gain_probabilities[index]
            weight = _prelec(cumulative, gamma) - _prelec(previous, gamma)
            utility += weight * gains[index]

    return utility


@njit(parallel=True, fastmath=True)
def sample_cpt(
    alpha: np.ndarray,
    lamda: np.ndarray,
    tau: np.ndarray,
    gamma: np.ndarray,
    outcomes_a: np.ndarray,
    outcomes_b: np.ndarray,
    probabilities_a: np.ndarray,
    probabilities_b: np.ndarray,
) -> dict[str, np.ndarray]:
    """Generate binary choices from a Cumulative Prospects Theory (CPT) model.

    Parameters
    ----------
    alpha : np.ndarray of shape (num_steps,)
        Curvature of the value function.
    lamda : np.ndarray of shape (num_steps,)
        Loss-aversion coefficient.
    tau : np.ndarray of shape (num_steps,)
        Choice sensitivity.
    gamma : np.ndarray of shape (num_steps,)
        Curvature of the Prelec probability-weighting function.
    outcomes_a, outcomes_b : ndarray, shape (num_trials, num_outcomes)
        Outcomes for options A and B. The two options may have different
        numbers of outcomes, but each must have at least one column.
    probabilities_a, probabilities_b : ndarray
        Probabilities corresponding to the outcome columns in A and B. Each
        row must be non-negative and sum to one.

    Returns
    -------
    data : dict of np.ndarray
        Named decision data. `"choice"` contains choices (1 for option A, 0 for option B).
    """
    if outcomes_a.ndim != 2 or outcomes_b.ndim != 2:
        raise ValueError("outcomes_a and outcomes_b must be two-dimensional")
    if probabilities_a.ndim != 2 or probabilities_b.ndim != 2:
        raise ValueError("probabilities_a and probabilities_b must be two-dimensional")
    if outcomes_a.shape != probabilities_a.shape or outcomes_b.shape != probabilities_b.shape:
        raise ValueError("Each outcome matrix must have the same shape as its probability matrix")
    if outcomes_a.shape[0] != outcomes_b.shape[0]:
        raise ValueError("outcomes_a and outcomes_b must have the same number of steps")

    probabilities_a = np.asarray(probabilities_a, dtype=np.float32)
    probabilities_b = np.asarray(probabilities_b, dtype=np.float32)

    if np.any(probabilities_a < 0.0) or np.any(probabilities_b < 0.0):
        raise ValueError("Probabilities must be non-negative")

    sums_a = probabilities_a.sum(axis=1)
    sums_b = probabilities_b.sum(axis=1)
    invalid_a = np.flatnonzero(~np.isclose(sums_a, 1.0, atol=1e-6, rtol=0.0))
    invalid_b = np.flatnonzero(~np.isclose(sums_b, 1.0, atol=1e-6, rtol=0.0))

    if invalid_a.size:
        raise ValueError("Probabilities_a must sum to 1 for every step")
    if invalid_b.size:
        raise ValueError("Probabilities_b must sum to 1 for every step")

    num_steps = lamda.shape[0]
    choices = np.empty(num_steps, dtype=np.int32)
    for i in prange(num_steps):
        alpha_t = alpha[i]
        lamda_t = lamda[i]
        tau_t = tau[i]
        gamma_t = gamma[i]

        utility_a = _cpt_utility(outcomes_a[i], probabilities_a[i], alpha_t, lamda_t, gamma_t)
        utility_b = _cpt_utility(outcomes_b[i], probabilities_b[i], alpha_t, lamda_t, gamma_t)
        logit = tau_t * (utility_a - utility_b)
        if logit >= 0.0:
            choice_probability = 1.0 / (1.0 + np.exp(-logit))
        else:
            exp_logit = np.exp(logit)
            choice_probability = exp_logit / (1.0 + exp_logit)
        choices[i] = np.random.binomial(1, choice_probability)

    return {"choice": choices}
