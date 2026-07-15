# User Guide

The core workflow has four pieces:

1. Define priors for static parameters with {class}`superstats.prior.Prior`.
2. Define time-varying dynamics with transition models such as
   {class}`superstats.transition.RandomWalk`,
   {class}`superstats.transition.AutoRegression`, or
   {class}`superstats.transition.GaussianProcess`.
3. Combine parameters in {class}`superstats.prior.JointPrior`.
4. Connect the prior and simulator through
   {class}`superstats.simulation.GenerativeModel` and train a
   {class}`superstats.workflow.Workflow`.

This page is the seed for the conceptual documentation. As the package grows,
the guide can expand into focused sections on transition models, simulators,
workflow training, and diagnostic interpretation.
