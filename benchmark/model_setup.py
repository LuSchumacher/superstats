"""Shared simulator setup for the benchmark.

This mirrors the DDM model configuration used in
`examples/benchmark.ipynb`, kept in one place so every benchmarked
network is trained/evaluated on an identical generative process.
"""

import superstats as sup


def build_generative_model() -> sup.simulation.GenerativeModel:
    """Build the joint prior + DDM simulator used throughout the benchmark.

    Returns
    -------
    generative_model : sup.simulation.GenerativeModel - the simulator
        shared across all benchmarked network parameterizations
    """
    joint_prior = sup.prior.JointPrior(
        v=sup.transition.RandomWalk(
            bounds=(0.0, 6.0),
            sigma=sup.prior.Prior(dist="halfnormal", scale=0.1),
            delta=0,
            initial_prior=sup.prior.Prior(dist="normal", loc=-1.5, scale=0.5),
        ),
        a=sup.transition.RandomWalk(
            bounds=(0.0, 4.0),
            sigma=sup.prior.Prior(dist="halfnormal", scale=0.1),
            delta=0,
            initial_prior=sup.prior.Prior(dist="normal", loc=0.0, scale=0.5),
        ),
        tau=sup.transition.RandomWalk(
            bounds=(0.0, 2.0),
            sigma=sup.prior.Prior(dist="halfnormal", scale=0.1),
            delta=0,
            initial_prior=sup.prior.Prior(dist="normal", loc=-1.5, scale=0.5),
        ),
        bias=0.5,
    )

    return sup.simulation.GenerativeModel(
        prior=joint_prior,
        model=sup.simulation.sample_ddm,
    )
