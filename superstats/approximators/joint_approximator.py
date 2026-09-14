"""Autoregressive smoothing and filtering with a global parameter head."""

from bayesflow.utils.serialization import serializable

from .composite_approximator import CompositeApproximator


@serializable("superstats.approximators")
class JointApproximator(CompositeApproximator):
    """Estimate an autoregressive sequence posterior and invariant posterior.

    Accepts the same arguments as MarginalApproximator, plus decoder_network.
    The internal sequence component is BayesFlow's AutoregressiveApproximator;
    the invariant component is a ContinuousApproximator. The shared observation
    encoder is owned only by this composite, keeping checkpoints serializable.

    Parameters
    ----------
    inference_network : bayesflow.networks.InferenceNetwork
        Conditional sequence density network.
    invariant_inference_network : bayesflow.networks.InferenceNetwork
        Separate density network for the invariant parameter vector.
    summary_network : keras.Layer, optional
        Observation encoder returning (B, T, H). See MarginalApproximator for
        its defaults and prefix evaluation in filtering mode.
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
        inference_network,
        invariant_inference_network,
        summary_network=None,
        decoder_network=None,
        invariant_pooling=None,
        adapter=None,
        mode="smoothing",
        standardize="all",
        **kwargs,
    ):
        super().__init__(
            inference_network=inference_network,
            invariant_inference_network=invariant_inference_network,
            summary_network=summary_network,
            decoder_network=decoder_network,
            invariant_pooling=invariant_pooling,
            adapter=adapter,
            mode=mode,
            standardize=standardize,
            joint=True,
            **kwargs,
        )

    @property
    def decoder_network(self):
        """The internally owned BayesFlow-compatible autoregressive decoder."""
        return self.sequence_approximator.decoder_network
