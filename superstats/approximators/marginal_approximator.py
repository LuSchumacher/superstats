"""Conditional marginal smoothing and filtering with a global parameter head."""

from bayesflow.utils.serialization import serializable

from .composite_approximator import CompositeApproximator


@serializable("superstats.approximators")
class MarginalApproximator(CompositeApproximator):
    """Estimate per-time posteriors conditional on inferred invariant parameters.

    Parameters
    ----------
    varying_inference_network : str or bayesflow.networks.InferenceNetwork, default: "coupling"
        Conditional per-time density network for varying parameters.
    invariant_inference_network : str or bayesflow.networks.InferenceNetwork, default: "coupling"
        Separate density network for the invariant parameter vector.
    summary_network : str or keras.Layer, default: "recurrent"
        Observation encoder returning (B, T, H). Defaults to a BayesFlow
        RecurrentNetwork, bidirectional for smoothing and causal for filtering.
    invariant_pooling : keras.Layer, optional
        Maps complete encoded observations (B, T, H) to (B, G). Defaults to
        masked mean pooling with log(1 + valid sequence length) appended.
    adapter : bayesflow.adapters.Adapter, optional
        Maps named inputs to separate inference_variables, invariant_variables,
        and summary_variables. Defaults to identity; see build_adapter.
    mode : {"smoothing", "filtering"}, optional
        Controls the observations available to each conditional sequence factor.
        The invariant head always uses all observations.
    standardize : str or sequence of str or None, optional
        "all" (default), or a subset of inference_variables, invariant_variables,
        summary_variables, inference_conditions. Each head standardizes its own
        targets; observed conditions are standardized before shared encoding.
    has_varying, has_invariant : bool, optional
        Enable each target group (both default to True). At least one is required.
        A custom network requires its corresponding target group.
    **kwargs
        Passed to the BayesFlow Approximator/Keras model base.

    Notes
    -----
    Sampling uses q(phi|x) * product_t q(theta_t|phi, x_available_at_t).
    The varying targets are conditionally independent across time given phi
    and observations. Shared sampled invariants still induce dependence.
    This is a marginal approximation, not a joint trajectory posterior.
    """

    def __init__(
        self,
        *,
        varying_inference_network="coupling",
        invariant_inference_network="coupling",
        summary_network="recurrent",
        invariant_pooling=None,
        adapter=None,
        mode="smoothing",
        standardize="all",
        has_varying=True,
        has_invariant=True,
        **kwargs,
    ):
        super().__init__(
            varying_inference_network=varying_inference_network,
            invariant_inference_network=invariant_inference_network,
            summary_network=summary_network,
            invariant_pooling=invariant_pooling,
            adapter=adapter,
            mode=mode,
            standardize=standardize,
            has_varying=has_varying,
            has_invariant=has_invariant,
            joint=False,
            **kwargs,
        )
