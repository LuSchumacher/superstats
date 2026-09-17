"""Autoregressive smoothing and filtering with a global parameter head."""

from bayesflow.utils.serialization import serializable

from .composite_approximator import CompositeApproximator


@serializable("superstats.approximators")
class JointApproximator(CompositeApproximator):
    """Estimate an autoregressive sequence posterior and invariant posterior.

    Accepts the same arguments as MarginalApproximator, plus decoder_network.
    The internal sequence component is BayesFlow's AutoregressiveApproximator;
    the invariant component is a ContinuousApproximator. The global summary
    network encodes observations once for both components. Its ready-made
    features are passed directly to the autoregressive decoder.

    Parameters
    ----------
    varying_inference_network : str or bayesflow.networks.InferenceNetwork, default: "coupling"
        Conditional sequence density network.
    invariant_inference_network : str or bayesflow.networks.InferenceNetwork, default: "coupling"
        Separate density network for the invariant parameter vector.
    summary_network : str or keras.Layer, default: "recurrent"
        Observation encoder returning (B, T, H). Defaults to a bidirectional
        RecurrentNetwork for smoothing or a unidirectional RecurrentNetwork
        for filtering.
    decoder_network : keras.Layer, optional
        BayesFlow-compatible decoder implementing call, compute_output_shape,
        initialize_cache, and decode_step. Defaults to TransformerDecoder for
        smoothing and RecurrentDecoder for filtering. Joint filtering requires
        RecurrentDecoder or a custom decoder declaring supports_filtering=True;
        that declaration promises no attention to future encoded positions.
    invariant_pooling : keras.Layer, optional
        Maps complete encoded observations (B, T, H) to (B, G). Defaults to
        masked mean pooling with log(1 + valid sequence length) appended.
    adapter : bayesflow.adapters.Adapter, optional
        Maps named inputs to the canonical separate target and observation keys.
    mode : {"smoothing", "filtering"}, optional
        Controls observation availability for the conditional sequence factors.
    standardize : str or sequence of str or None, optional
        "all" (default), or canonical variable keys as in MarginalApproximator.
    has_varying, has_invariant : bool, optional
        Enable each target group (both default to True). At least one is required.
        A custom network requires its corresponding target group.
    **kwargs
        Passed to the BayesFlow Approximator/Keras model base.

    Notes
    -----
    Smoothing uses q(phi|x_1:T) * product_t
    q(theta_t|theta_<t, phi, x_1:T).
    Filtering uses q(phi|x_1:T) * product_t
    q(theta_t|theta_<t, phi, x_1:t). These are conditional filtering factors;
    integrating full-sequence invariant draws is not strict online filtering.
    """

    def __init__(
        self,
        *,
        varying_inference_network="coupling",
        invariant_inference_network="coupling",
        summary_network="recurrent",
        decoder_network=None,
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
            decoder_network=decoder_network,
            invariant_pooling=invariant_pooling,
            adapter=adapter,
            mode=mode,
            standardize=standardize,
            has_varying=has_varying,
            has_invariant=has_invariant,
            joint=True,
            **kwargs,
        )

    @property
    def decoder_network(self):
        """The internally owned BayesFlow-compatible autoregressive decoder."""
        return self.sequence_approximator.decoder_network if self.has_varying else None
