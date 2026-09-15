"""Dispatch helpers for workflow and generative-model construction."""

from functools import singledispatch

import bayesflow as bf
import keras

from superstats.defaults import (
    DEFAULT_COUPLING_FLOW,
    DEFAULT_RECURRENT_NETWORK,
    DEFAULT_TRANSFORMER_NETWORK,
)


def _merge_defaults(defaults, kwargs):
    return {**defaults, **kwargs}


@singledispatch
def find_approximator(arg, *args, **kwargs):
    raise TypeError(f"approximator must be 'marginal', 'joint', an approximator instance, or None, not {arg!r}.")


@find_approximator.register
def _(arg: None, *args, **kwargs):
    return None


@find_approximator.register
def _(name: str, *args, summary_network=None, inference_network=None, invariant_inference_network=None, **kwargs):
    from superstats.approximators import JointApproximator, MarginalApproximator

    match name.lower():
        case "marginal":
            constructor = MarginalApproximator
        case "joint":
            constructor = JointApproximator
        case unknown_approximator:
            raise ValueError(f"Unknown approximator: {unknown_approximator!r}.")

    summary_network = None if summary_network is None else find_embedding_network(summary_network)
    inference_network = None if inference_network is None else find_inference_network(inference_network)
    invariant_inference_network = (
        None if invariant_inference_network is None else find_inference_network(invariant_inference_network)
    )

    return constructor(
        *args,
        summary_network=summary_network,
        inference_network=inference_network,
        invariant_inference_network=invariant_inference_network,
        **kwargs,
    )


@find_approximator.register
def _(approximator: bf.approximators.Approximator, *args, **kwargs):
    return approximator


@singledispatch
def find_embedding_network(arg, *args, **kwargs):
    raise TypeError(
        f"embedding network must be one of 'recurrent', 'transformer', or a keras.Layer instance, not {arg!r}."
    )


@find_embedding_network.register
def _(name: str, *args, **kwargs):
    match name.lower():
        case "recurrent":
            return bf.networks.RecurrentNetwork(*args, **_merge_defaults(DEFAULT_RECURRENT_NETWORK, kwargs))
        case "transformer":
            return bf.networks.TimeSeriesTransformer(
                *args,
                **_merge_defaults(DEFAULT_TRANSFORMER_NETWORK, kwargs),
            )
        case unknown_network:
            raise ValueError(f"Unknown embedding network: {unknown_network!r}.")


@find_embedding_network.register
def _(network: keras.Layer, *args, **kwargs):
    return network


@singledispatch
def find_inference_network(arg, *args, **kwargs):
    raise TypeError(
        f"inference_network must be one of 'coupling', 'coupling_flow' or a keras.Layer instance, not {arg!r}."
    )


@find_inference_network.register
def _(name: str, *args, **kwargs):
    match name.lower():
        case "coupling" | "coupling_flow":
            return bf.networks.CouplingFlow(*args, **_merge_defaults(DEFAULT_COUPLING_FLOW, kwargs))
        case unknown_network:
            raise ValueError(f"Unknown inference network: {unknown_network!r}.")


@find_inference_network.register
def _(network: keras.Layer, *args, **kwargs):
    return network


@singledispatch
def find_missing(arg, *args, **kwargs):
    if callable(arg):
        return arg
    raise TypeError("missing must be None, 'random', a MissingProcess instance, or callable")


@find_missing.register
def _(arg: None, *args, **kwargs):
    return None


@find_missing.register
def _(name: str, *args, **kwargs):
    match name.lower():
        case "random":
            from superstats.simulation.augmentation.random_missing import RandomMissingProcess

            return RandomMissingProcess(*args, **kwargs)
        case _:
            raise TypeError("missing must be None, 'random', a MissingProcess instance, or callable")


@singledispatch
def find_contamination(arg, *args, **kwargs):
    if callable(arg):
        return arg
    raise TypeError("contamination must be None, 'random_choice', a ContaminationProcess instance, or callable")


@find_contamination.register
def _(arg: None, *args, **kwargs):
    return None


@find_contamination.register
def _(name: str, *args, **kwargs):
    match name.lower():
        case "random_choice":
            from superstats.simulation.augmentation.random_choice_contamination import RandomChoiceContamination

            return RandomChoiceContamination(*args, **kwargs)
        case _:
            raise TypeError("contamination must be None, 'random_choice', a ContaminationProcess instance, or callable")
