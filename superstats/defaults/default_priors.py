from superstats.prior.prior import Prior

DEFAULT_GLOBAL_PRIORS = {
    "sigma_prior": Prior("halfnormal", scale = 0.1),
    "delta_prior": Prior("normal", loc = 0.0, scale = 0.05),
    "phi_prior": Prior("beta", a = 20.0, b = 1.0),
}