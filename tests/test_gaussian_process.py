import numpy as np
import pytest

from superstats.prior import Prior
from superstats.transition import GaussianProcess
from superstats.transition.gaussian_process import sample_gaussian_process
from superstats.transition.kernel import (
    CompositeKernel,
    Kernel,
    LinearKernel,
    PeriodicKernel,
    RBFKernel,
    build_abs_dist_mat,
    build_sq_dist_mat,
    get_linear_kernel,
    get_periodic_kernel,
    get_rbf_kernel,
    resolve_kernel,
)

BATCH_SIZE = 8
NUM_STEPS = 12


def test_build_sq_dist_mat_shape_and_diagonal():
    sq_dist = build_sq_dist_mat(NUM_STEPS)
    assert sq_dist.shape == (NUM_STEPS, NUM_STEPS)
    assert np.allclose(np.diag(sq_dist), 0.0)
    assert np.allclose(sq_dist, sq_dist.T)
    off_diag = sq_dist[~np.eye(NUM_STEPS, dtype=bool)]
    assert np.all(off_diag > 0)


def test_build_sq_dist_mat_domain_is_unit_interval():
    # domain must be [0, 1] regardless of num_steps, not num_steps-dependent
    sq_dist_short = build_sq_dist_mat(10)
    sq_dist_long = build_sq_dist_mat(1000)
    assert np.isclose(sq_dist_short.max(), 1.0, atol=1e-2)
    assert np.isclose(sq_dist_long.max(), 1.0, atol=1e-2)


def test_build_abs_dist_mat_shape_and_diagonal():
    abs_dist = build_abs_dist_mat(NUM_STEPS)
    assert abs_dist.shape == (NUM_STEPS, NUM_STEPS)
    assert np.allclose(np.diag(abs_dist), 0.0)
    assert np.allclose(abs_dist, abs_dist.T)
    off_diag = abs_dist[~np.eye(NUM_STEPS, dtype=bool)]
    assert np.all(off_diag > 0)
    # abs_dist is the elementwise sqrt of sq_dist
    assert np.allclose(abs_dist, np.sqrt(build_sq_dist_mat(NUM_STEPS)))


def test_get_rbf_kernel_batched_shape_symmetric_psd():
    length_scale = np.full(BATCH_SIZE, 0.2)
    amplitude = np.full(BATCH_SIZE, 1.0)

    kernel_mat = get_rbf_kernel(NUM_STEPS, length_scale, amplitude)

    assert kernel_mat.shape == (BATCH_SIZE, NUM_STEPS, NUM_STEPS)
    for b in range(BATCH_SIZE):
        assert np.allclose(kernel_mat[b], kernel_mat[b].T)
        np.linalg.cholesky(kernel_mat[b] + 1e-8 * np.eye(NUM_STEPS))
    assert np.allclose(np.diagonal(kernel_mat, axis1=1, axis2=2), 1.0)


def test_get_rbf_kernel_amplitude_scales_variance():
    length_scale = np.full(BATCH_SIZE, 0.2)
    small_amp = get_rbf_kernel(NUM_STEPS, length_scale, np.full(BATCH_SIZE, 1.0))
    large_amp = get_rbf_kernel(NUM_STEPS, length_scale, np.full(BATCH_SIZE, 2.0))
    assert np.allclose(large_amp, 4.0 * small_amp)


def test_get_rbf_kernel_length_scale_controls_decay():
    short = get_rbf_kernel(NUM_STEPS, np.full(BATCH_SIZE, 0.02), np.full(BATCH_SIZE, 1.0))
    long = get_rbf_kernel(NUM_STEPS, np.full(BATCH_SIZE, 5.0), np.full(BATCH_SIZE, 1.0))
    off_diag_short = short[0][~np.eye(NUM_STEPS, dtype=bool)]
    off_diag_long = long[0][~np.eye(NUM_STEPS, dtype=bool)]
    assert off_diag_long.mean() > off_diag_short.mean()


def test_get_linear_kernel_batched_shape_symmetric():
    variance = np.full(BATCH_SIZE, 1.0)
    kernel_mat = get_linear_kernel(NUM_STEPS, variance)

    assert kernel_mat.shape == (BATCH_SIZE, NUM_STEPS, NUM_STEPS)
    for b in range(BATCH_SIZE):
        assert np.allclose(kernel_mat[b], kernel_mat[b].T)


def test_get_linear_kernel_variance_scales_linearly():
    base = get_linear_kernel(NUM_STEPS, np.full(BATCH_SIZE, 1.0))
    scaled = get_linear_kernel(NUM_STEPS, np.full(BATCH_SIZE, 3.0))
    assert np.allclose(scaled, 3.0 * base)


def test_get_linear_kernel_zero_variance_at_origin():
    # domain starts at x=0, so variance must be exactly zero at t=0
    # regardless of the variance hyperparameter (c=0 pinning point)
    variance = np.full(BATCH_SIZE, 5.0)
    kernel_mat = get_linear_kernel(NUM_STEPS, variance)
    assert np.allclose(kernel_mat[:, 0, 0], 0.0)


def test_get_periodic_kernel_batched_shape_symmetric_psd():
    length_scale = np.full(BATCH_SIZE, 0.5)
    period = np.full(BATCH_SIZE, 0.25)
    amplitude = np.full(BATCH_SIZE, 1.0)

    kernel_mat = get_periodic_kernel(NUM_STEPS, length_scale, period, amplitude)

    assert kernel_mat.shape == (BATCH_SIZE, NUM_STEPS, NUM_STEPS)
    for b in range(BATCH_SIZE):
        assert np.allclose(kernel_mat[b], kernel_mat[b].T)
        np.linalg.cholesky(kernel_mat[b] + 1e-6 * np.eye(NUM_STEPS))
    assert np.allclose(np.diagonal(kernel_mat, axis1=1, axis2=2), 1.0)


def test_get_periodic_kernel_amplitude_scales_variance():
    length_scale = np.full(BATCH_SIZE, 0.5)
    period = np.full(BATCH_SIZE, 0.25)
    small_amp = get_periodic_kernel(NUM_STEPS, length_scale, period, np.full(BATCH_SIZE, 1.0))
    large_amp = get_periodic_kernel(NUM_STEPS, length_scale, period, np.full(BATCH_SIZE, 2.0))
    assert np.allclose(large_amp, 4.0 * small_amp)


def test_get_periodic_kernel_recurs_at_multiples_of_period():
    # fine grid so a full period lands close to a grid point
    num_steps = 400
    length_scale = np.full(1, 0.5)
    period = np.full(1, 0.2)
    amplitude = np.full(1, 1.0)

    kernel_mat = get_periodic_kernel(num_steps, length_scale, period, amplitude)
    x = np.linspace(0, 1, num_steps)

    idx_period = np.argmin(np.abs(x - period[0]))
    idx_half_period = np.argmin(np.abs(x - period[0] / 2))

    k_00 = kernel_mat[0, 0, 0]
    k_0_period = kernel_mat[0, 0, idx_period]
    k_0_half_period = kernel_mat[0, 0, idx_half_period]

    # correlation near a full period should be close to the self-correlation
    assert k_0_period == pytest.approx(k_00, abs=0.05)
    # correlation at a half period should be substantially lower
    assert k_0_half_period < 0.5 * k_00


def test_get_periodic_kernel_length_scale_controls_decay_within_period():
    period = np.full(BATCH_SIZE, 0.5)
    short = get_periodic_kernel(NUM_STEPS, np.full(BATCH_SIZE, 0.1), period, np.full(BATCH_SIZE, 1.0))
    long = get_periodic_kernel(NUM_STEPS, np.full(BATCH_SIZE, 2.0), period, np.full(BATCH_SIZE, 1.0))
    off_diag_short = short[0][~np.eye(NUM_STEPS, dtype=bool)]
    off_diag_long = long[0][~np.eye(NUM_STEPS, dtype=bool)]
    assert off_diag_long.mean() > off_diag_short.mean()


def test_sample_gaussian_process_preserves_float32_dtype():
    length_scale = np.full(BATCH_SIZE, 0.3, dtype=np.float32)
    amplitude = np.full(BATCH_SIZE, 1.0, dtype=np.float32)
    kernel_mat = get_rbf_kernel(NUM_STEPS, length_scale, amplitude)

    start = np.zeros(BATCH_SIZE, dtype=np.float32)
    local_params = np.empty((BATCH_SIZE, NUM_STEPS), dtype=np.float32)
    bounds = np.array([0.0, 1.0], dtype=np.float32)

    out = sample_gaussian_process(local_params, start, kernel_mat, bounds)

    assert out.dtype == np.float32
    assert local_params.dtype == np.float32
    # in-place fill: returned array is the same buffer that was passed in
    assert out is local_params


def test_sample_gaussian_process_respects_bounds():
    length_scale = np.full(BATCH_SIZE, 0.1)
    amplitude = np.full(BATCH_SIZE, 3.0)  # large amplitude, would overshoot without squashing
    kernel_mat = get_rbf_kernel(NUM_STEPS, length_scale, amplitude)

    start = np.zeros(BATCH_SIZE)
    local_params = np.empty((BATCH_SIZE, NUM_STEPS))
    bounds = np.array([-1.0, 2.0])

    out = sample_gaussian_process(local_params, start, kernel_mat, bounds)

    assert np.all(out >= -1.0 - 1e-6)
    assert np.all(out <= 2.0 + 1e-6)


def test_rbf_kernel_default_hyperparam_names_unprefixed():
    kernel = RBFKernel()
    assert kernel.hyperparam_names == ("length_scale", "amplitude")


def test_linear_kernel_default_hyperparam_names_unprefixed():
    kernel = LinearKernel()
    assert kernel.hyperparam_names == ("variance",)


def test_periodic_kernel_default_hyperparam_names_unprefixed():
    kernel = PeriodicKernel()
    assert kernel.hyperparam_names == ("length_scale", "period", "amplitude")


def test_kernel_custom_name_prefixes_hyperparam_names():
    kernel = RBFKernel(name="trend")
    assert kernel.hyperparam_names == ("trend_length_scale", "trend_amplitude")


def test_periodic_kernel_custom_name_prefixes_hyperparam_names():
    kernel = PeriodicKernel(name="season")
    assert kernel.hyperparam_names == ("season_length_scale", "season_period", "season_amplitude")


def test_rbf_kernel_build_matches_low_level_function():
    kernel = RBFKernel()
    length_scale = np.full(BATCH_SIZE, 0.3)
    amplitude = np.full(BATCH_SIZE, 1.5)

    built = kernel.build(NUM_STEPS, length_scale=length_scale, amplitude=amplitude)
    expected = get_rbf_kernel(NUM_STEPS, length_scale, amplitude)

    assert np.allclose(built, expected)


def test_rbf_kernel_build_with_custom_name_uses_prefixed_kwargs():
    kernel = RBFKernel(name="trend")
    length_scale = np.full(BATCH_SIZE, 0.3)
    amplitude = np.full(BATCH_SIZE, 1.5)

    built = kernel.build(NUM_STEPS, trend_length_scale=length_scale, trend_amplitude=amplitude)
    expected = get_rbf_kernel(NUM_STEPS, length_scale, amplitude)

    assert np.allclose(built, expected)


def test_linear_kernel_build_matches_low_level_function():
    kernel = LinearKernel()
    variance = np.full(BATCH_SIZE, 2.0)

    built = kernel.build(NUM_STEPS, variance=variance)
    expected = get_linear_kernel(NUM_STEPS, variance)

    assert np.allclose(built, expected)


def test_periodic_kernel_build_matches_low_level_function():
    kernel = PeriodicKernel()
    length_scale = np.full(BATCH_SIZE, 0.4)
    period = np.full(BATCH_SIZE, 0.2)
    amplitude = np.full(BATCH_SIZE, 1.2)

    built = kernel.build(NUM_STEPS, length_scale=length_scale, period=period, amplitude=amplitude)
    expected = get_periodic_kernel(NUM_STEPS, length_scale, period, amplitude)

    assert np.allclose(built, expected)


def test_periodic_kernel_build_with_custom_name_uses_prefixed_kwargs():
    kernel = PeriodicKernel(name="season")
    length_scale = np.full(BATCH_SIZE, 0.4)
    period = np.full(BATCH_SIZE, 0.2)
    amplitude = np.full(BATCH_SIZE, 1.2)

    built = kernel.build(
        NUM_STEPS,
        season_length_scale=length_scale,
        season_period=period,
        season_amplitude=amplitude,
    )
    expected = get_periodic_kernel(NUM_STEPS, length_scale, period, amplitude)

    assert np.allclose(built, expected)


def test_composite_kernel_add_sums_matrices():
    rbf = RBFKernel()
    linear = LinearKernel()
    composite = rbf + linear

    assert isinstance(composite, CompositeKernel)
    assert isinstance(composite, Kernel)
    assert composite.hyperparam_names == ("length_scale", "amplitude", "variance")

    hyperparams = {
        "length_scale": np.full(BATCH_SIZE, 0.3),
        "amplitude": np.full(BATCH_SIZE, 1.0),
        "variance": np.full(BATCH_SIZE, 1.0),
    }

    built = composite.build(NUM_STEPS, **hyperparams)
    expected = get_rbf_kernel(NUM_STEPS, hyperparams["length_scale"], hyperparams["amplitude"]) + get_linear_kernel(
        NUM_STEPS, hyperparams["variance"]
    )
    assert np.allclose(built, expected)


def test_composite_kernel_mul_multiplies_matrices():
    composite = RBFKernel() * LinearKernel()

    hyperparams = {
        "length_scale": np.full(BATCH_SIZE, 0.3),
        "amplitude": np.full(BATCH_SIZE, 1.0),
        "variance": np.full(BATCH_SIZE, 1.0),
    }

    built = composite.build(NUM_STEPS, **hyperparams)
    expected = get_rbf_kernel(NUM_STEPS, hyperparams["length_scale"], hyperparams["amplitude"]) * get_linear_kernel(
        NUM_STEPS, hyperparams["variance"]
    )
    assert np.allclose(built, expected)


def test_composite_kernel_rbf_times_periodic_locally_periodic():
    # the classic "locally periodic" kernel: RBF envelope * periodic pattern
    composite = RBFKernel(name="envelope") * PeriodicKernel(name="season")

    assert composite.hyperparam_names == (
        "envelope_length_scale",
        "envelope_amplitude",
        "season_length_scale",
        "season_period",
        "season_amplitude",
    )

    hyperparams = {
        "envelope_length_scale": np.full(BATCH_SIZE, 0.5),
        "envelope_amplitude": np.full(BATCH_SIZE, 1.0),
        "season_length_scale": np.full(BATCH_SIZE, 0.3),
        "season_period": np.full(BATCH_SIZE, 0.2),
        "season_amplitude": np.full(BATCH_SIZE, 1.0),
    }

    built = composite.build(NUM_STEPS, **hyperparams)
    for b in range(BATCH_SIZE):
        np.linalg.cholesky(built[b] + 1e-6 * np.eye(NUM_STEPS))


def test_composite_kernel_of_same_type_without_names_raises():
    with pytest.raises(ValueError):
        RBFKernel() + RBFKernel()


def test_composite_kernel_of_same_periodic_type_without_names_raises():
    with pytest.raises(ValueError):
        PeriodicKernel() + PeriodicKernel()


def test_composite_kernel_of_same_type_with_names_succeeds():
    composite = RBFKernel(name="short") + RBFKernel(name="long")
    assert composite.hyperparam_names == (
        "short_length_scale",
        "short_amplitude",
        "long_length_scale",
        "long_amplitude",
    )


def test_composite_kernel_invalid_op_raises():
    with pytest.raises(ValueError):
        CompositeKernel(RBFKernel(name="a"), LinearKernel(), op="xor")


def test_composite_kernel_nesting_aggregates_names():
    nested = (RBFKernel(name="trend") + LinearKernel()) * RBFKernel(name="mod")
    assert set(nested.hyperparam_names) == {
        "trend_length_scale",
        "trend_amplitude",
        "variance",
        "mod_length_scale",
        "mod_amplitude",
    }


def test_resolve_kernel_from_string():
    assert isinstance(resolve_kernel("rbf"), RBFKernel)
    assert isinstance(resolve_kernel("linear"), LinearKernel)
    assert isinstance(resolve_kernel("periodic"), PeriodicKernel)


def test_resolve_kernel_passthrough_instance():
    kernel = RBFKernel(name="custom")
    assert resolve_kernel(kernel) is kernel


def test_resolve_kernel_unknown_string_raises():
    with pytest.raises(ValueError):
        resolve_kernel("matern")


def test_resolve_kernel_invalid_type_raises():
    with pytest.raises(TypeError):
        resolve_kernel(123)


@pytest.mark.parametrize(
    "gp, expected_hyper_keys, expected_fixed_keys",
    [
        (GaussianProcess(), {"length_scale"}, {"amplitude"}),
        (GaussianProcess(kernel="linear"), {"variance"}, set()),
        (GaussianProcess(kernel="periodic"), {"length_scale", "period"}, {"amplitude"}),
        (
            GaussianProcess(kernel=RBFKernel() + LinearKernel()),
            {"length_scale", "variance"},
            {"amplitude"},
        ),
    ],
)
def test_gaussian_process_sample_shape_and_keys(gp, expected_hyper_keys, expected_fixed_keys):
    result = gp.sample(batch_size=BATCH_SIZE, num_steps=NUM_STEPS)

    assert set(result.keys()) == {"local_params", "hyper_params", "fixed_params"}

    local_params = result["local_params"]
    assert isinstance(local_params, np.ndarray)
    assert local_params.shape == (BATCH_SIZE, NUM_STEPS)
    assert local_params.dtype == np.float32

    lower, upper = gp.bounds
    assert np.all(local_params >= lower - 1e-4)
    assert np.all(local_params <= upper + 1e-4)

    assert set(result["hyper_params"].keys()) == expected_hyper_keys
    for values in result["hyper_params"].values():
        assert values.shape == (BATCH_SIZE,)

    assert set(result["fixed_params"].keys()) == expected_fixed_keys


def test_gaussian_process_default_amplitude_is_fixed_at_one():
    gp = GaussianProcess()
    result = gp.sample(batch_size=BATCH_SIZE, num_steps=NUM_STEPS)
    assert result["fixed_params"]["amplitude"] == 1.0


def test_gaussian_process_custom_bounds_respected():
    gp = GaussianProcess(bounds=(-2.0, 5.0))
    result = gp.sample(batch_size=BATCH_SIZE, num_steps=NUM_STEPS)
    local_params = result["local_params"]
    assert np.all(local_params >= -2.0 - 1e-4)
    assert np.all(local_params <= 5.0 + 1e-4)


def test_gaussian_process_kernel_params_override_default_prior():
    gp = GaussianProcess(kernel_params={"length_scale": 0.1, "amplitude": 2.0})
    result = gp.sample(batch_size=BATCH_SIZE, num_steps=NUM_STEPS)
    assert result["hyper_params"] == {}
    assert result["fixed_params"] == {"length_scale": 0.1, "amplitude": 2.0}


def test_gaussian_process_periodic_kernel_params_override_default_prior():
    gp = GaussianProcess(
        kernel="periodic",
        kernel_params={"length_scale": 0.3, "period": 0.25, "amplitude": 1.0},
    )
    result = gp.sample(batch_size=BATCH_SIZE, num_steps=NUM_STEPS)
    assert result["hyper_params"] == {}
    assert result["fixed_params"] == {"length_scale": 0.3, "period": 0.25, "amplitude": 1.0}


def test_gaussian_process_unknown_kernel_param_raises():
    with pytest.raises(ValueError):
        GaussianProcess(kernel_params={"not_a_real_param": 1.0})


def test_gaussian_process_unknown_kernel_string_raises():
    with pytest.raises(ValueError):
        GaussianProcess(kernel="matern")


def test_gaussian_process_composite_kernel_custom_names_requires_explicit_params():
    kernel = RBFKernel(name="trend") + RBFKernel(name="local")
    with pytest.raises(KeyError):
        GaussianProcess(kernel=kernel).sample(batch_size=BATCH_SIZE, num_steps=NUM_STEPS)


def test_gaussian_process_composite_kernel_custom_names_with_explicit_params():
    kernel = RBFKernel(name="trend") + RBFKernel(name="local")
    gp = GaussianProcess(
        kernel=kernel,
        kernel_params={
            "trend_length_scale": Prior("halfnormal", scale=1.5),
            "trend_amplitude": 1.0,
            "local_length_scale": Prior("halfnormal", scale=0.1),
            "local_amplitude": 1.0,
        },
        initial_prior=Prior("normal", loc=0.0, scale=1.0),
    )
    result = gp.sample(batch_size=BATCH_SIZE, num_steps=NUM_STEPS)

    assert set(result["hyper_params"].keys()) == {"trend_length_scale", "local_length_scale"}
    assert set(result["fixed_params"].keys()) == {"trend_amplitude", "local_amplitude"}
    assert result["local_params"].shape == (BATCH_SIZE, NUM_STEPS)


def test_gaussian_process_locally_periodic_kernel():
    kernel = RBFKernel(name="envelope") * PeriodicKernel(name="season")
    gp = GaussianProcess(
        kernel=kernel,
        kernel_params={
            "envelope_length_scale": Prior("halfnormal", scale=0.5),
            "envelope_amplitude": 1.0,
            "season_length_scale": Prior("halfnormal", scale=0.3),
            "season_period": Prior("uniform", low=0.05, high=0.5),
            "season_amplitude": 1.0,
        },
    )
    result = gp.sample(batch_size=BATCH_SIZE, num_steps=NUM_STEPS)

    assert set(result["hyper_params"].keys()) == {
        "envelope_length_scale",
        "season_length_scale",
        "season_period",
    }
    assert set(result["fixed_params"].keys()) == {"envelope_amplitude", "season_amplitude"}
    assert np.all(np.isfinite(result["local_params"]))


def test_gaussian_process_accepts_kernel_instance_directly():
    gp = GaussianProcess(kernel=LinearKernel())
    result = gp.sample(batch_size=BATCH_SIZE, num_steps=NUM_STEPS)
    assert set(result["hyper_params"].keys()) == {"variance"}


def test_gaussian_process_is_stochastic_across_batch_elements():
    gp = GaussianProcess()
    result = gp.sample(batch_size=BATCH_SIZE, num_steps=NUM_STEPS)
    local_params = result["local_params"]
    assert not np.allclose(local_params[0], local_params[1])


def test_gaussian_process_finite_values_for_many_steps():
    gp = GaussianProcess()
    result = gp.sample(batch_size=4, num_steps=200)
    assert np.all(np.isfinite(result["local_params"]))


def test_gaussian_process_periodic_finite_values_for_many_steps():
    gp = GaussianProcess(kernel="periodic")
    result = gp.sample(batch_size=4, num_steps=200)
    assert np.all(np.isfinite(result["local_params"]))
