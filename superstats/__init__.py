# from superstats.priors import JointPrior, Prior
# from superstats.transition import AR
# from superstats.simulation import Simulator
# from superstats.study import Study


# prior = JointPrior(
#     v=AR(..., initial_prior=Prior("normal", loc=0, scale=3.), bounds=[0., 3.], transformed=True, sigma_prior=Prior()),
#     a=AR(..., initial_prior=Prior("normal", loc=0.2, scale=3.), bounds=[0., 3.], transformed=True),
#     ndt=Prior("normal", loc=0.2, scale=3.)
# )


# samples = prior.sample(time=100)

# def sim(v, a, ndt):
#     """user defined"""

#     pass


# # bf.make_simulator([prior, sim], paralell=True)
# dynamic_model = Simulator(prior=prior, simulator=sim)

# study = Study(model=model, estimator="flow_matching", inference="smoothing")

# history = study.train(**kwargs, optimize=True)

# diagnostics = study.diagnose(**kwargs)

# posteriors = study.estimate(real_data_dict)


# from diffusion import DDM

# simulator = DDM(v="ar", a="static", ndt="static")
