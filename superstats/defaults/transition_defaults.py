from superstats.prior.prior import Prior

DEFAULT_HYPER_PRIORS = {
    "sigma": Prior("halfnormal", scale=0.1),
    "phi": Prior("beta", a=20.0, b=1.0),
    "theta": Prior("halfnormal", scale=0.3),
    "mu": Prior("normal", loc=0.0, scale=1.0),
}

DEFAULT_BOUNDS = (0.0, 1.0)

DEFAULT_INITIAL_PRIOR = Prior("normal", loc=0.0, scale=1.0)
