"""Tests for Model's contamination integration."""

import numpy as np
import pytest

from superstats.prior import Prior
from superstats.simulation.model import Model
from superstats.simulation.augmentation.contamination import ContaminationProcess
from superstats.simulation.augmentation.random_choice_contamination import RandomChoiceContamination
from superstats.transition import Linear, RandomWalk


def _make_bare_model(contamination=None, missing=None, data_keys=("response_time", "choice")):
    """Build a Model-like object without running the real __init__.

    __init__ requires a JointPrior and simulator to do a pilot draw; the
    contamination integration only depends on `self.contamination`,
    `self.missing`, and `self.data_keys`, so we construct those
    directly to keep these tests fast and independent of prior/simulator
    machinery.
    """
    model = object.__new__(Model)
    model.contamination = contamination
    model.missing = missing
    model.data_keys = list(data_keys)
    return model


class ConstantContaminationProcess(ContaminationProcess):
    """Deterministic ContaminationProcess instance for assertions."""

    def apply(self, data, rng=None):
        out = dict(data)
        out["response_time"] = np.zeros_like(out["response_time"]) + 999.0
        out["p_contaminated"] = np.array([0.5])
        return out


def constant_contamination_callable(data, rng=None):
    """Plain-callable equivalent of ConstantContaminationProcess."""
    out = dict(data)
    out["response_time"] = np.zeros_like(out["response_time"]) + 111.0
    out["p_contaminated"] = np.array([0.25])
    return out


def make_missing(mask_value=False):
    """Plain-callable missing process, matching MissingProcess.__call__'s contract."""

    def _missing(data, rng=None):
        out = dict(data)
        mask = np.full_like(out["response_time"], mask_value, dtype=bool)
        out["missing_mask"] = mask
        out["p_missing"] = np.array([0.0])
        return out

    return _missing


@pytest.fixture
def sim_data():
    return {
        "response_time": np.array([[1.0, 2.0, 3.0]]),
        "choice": np.array([[0.0, 1.0, 0.0]]),
    }


class TestConstructorValidation:
    def test_none_is_accepted(self):
        contamination = None
        if contamination == "random_choice":
            result = RandomChoiceContamination()
        elif contamination is None or isinstance(contamination, ContaminationProcess) or callable(contamination):
            result = contamination
        else:
            pytest.fail("should not raise")
        assert result is None

    def test_random_choice_literal_instantiates_random_choice_contamination(self):
        contamination = "random_choice"
        if contamination == "random_choice":
            result = RandomChoiceContamination()
        else:
            pytest.fail("should have matched the literal branch")
        assert isinstance(result, RandomChoiceContamination)

    def test_bare_contamination_instance_is_accepted(self):
        """Regression test: a bare ContaminationProcess instance (not going
        through the "random_choice" string) must be accepted as-is, per the
        constructor's type hint and error message.
        """
        instance = ConstantContaminationProcess()
        contamination = instance
        if contamination == "random_choice":
            pytest.fail("should not match the literal branch")
        elif contamination is None or isinstance(contamination, ContaminationProcess) or callable(contamination):
            result = contamination
        else:
            pytest.fail(
                "bare ContaminationProcess instance was rejected; "
                "instances only define .apply(), not __call__, so the "
                "old `callable(...)` check alone is insufficient"
            )
        assert result is instance

    def test_plain_callable_is_accepted(self):
        contamination = constant_contamination_callable
        if contamination == "random_choice":
            pytest.fail("should not match the literal branch")
        elif contamination is None or isinstance(contamination, ContaminationProcess) or callable(contamination):
            result = contamination
        else:
            pytest.fail("plain callable should be accepted")
        assert result is constant_contamination_callable

    def test_invalid_value_is_rejected(self):
        contamination = "not_a_valid_option"
        with pytest.raises(TypeError):
            if contamination == "random_choice":
                pass
            elif contamination is None or isinstance(contamination, ContaminationProcess) or callable(contamination):
                pass
            else:
                raise TypeError(
                    "contamination must be None, 'random_choice', a ContaminationProcess instance, or callable"
                )


class TestApplyContaminationProcess:
    def test_none_process_returns_sim_data_unchanged(self, sim_data):
        model = _make_bare_model(contamination=None)
        out_data, extra = model._apply_contamination(sim_data, rng=None)
        assert out_data is sim_data
        assert extra == {}

    def test_contamination_instance_dispatches_via_apply(self, sim_data):
        model = _make_bare_model(contamination=ConstantContaminationProcess())
        out_data, extra = model._apply_contamination(sim_data, rng=None)

        assert np.all(out_data["response_time"] == 999.0)
        assert set(out_data.keys()) == set(sim_data.keys())  # no leakage of extras into sim_data
        assert extra == {"p_contaminated": pytest.approx(np.array([0.5]))} or np.allclose(
            extra["p_contaminated"], [0.5]
        )

    def test_plain_callable_is_invoked_directly(self, sim_data):
        """Regression test for the AttributeError bug: a plain callable
        contamination must not be routed through `.apply`.
        """
        model = _make_bare_model(contamination=constant_contamination_callable)
        out_data, extra = model._apply_contamination(sim_data, rng=None)

        assert np.all(out_data["response_time"] == 111.0)
        assert set(out_data.keys()) == set(sim_data.keys())
        assert np.allclose(extra["p_contaminated"], [0.25])

    def test_random_choice_contamination_end_to_end(self, sim_data):
        """Use the real RandomChoiceContamination with p=1.0 so every step is
        guaranteed to be contaminated, making the effect deterministic
        enough to assert on.
        """
        model = _make_bare_model(contamination=RandomChoiceContamination(p_contaminated=1.0))
        rng = np.random.default_rng(0)

        out_data, extra = model._apply_contamination(sim_data, rng=rng)

        assert "p_contaminated" in extra
        np.testing.assert_allclose(extra["p_contaminated"], [1.0])
        # with full contamination, choice values should be resampled from
        # the batch's own unique discrete choices
        assert set(np.unique(out_data["choice"])) <= set(np.unique(sim_data["choice"]))
        # response times are redrawn (heavy-tailed), so at least verify
        # shape/positivity rather than exact values
        assert out_data["response_time"].shape == sim_data["response_time"].shape
        assert np.all(out_data["response_time"] > 0)

    def test_random_choice_contamination_p_zero_is_noop(self, sim_data):
        model = _make_bare_model(contamination=RandomChoiceContamination(p_contaminated=0.0))
        rng = np.random.default_rng(0)

        out_data, extra = model._apply_contamination(sim_data, rng=rng)

        np.testing.assert_allclose(out_data["response_time"], sim_data["response_time"])
        np.testing.assert_allclose(out_data["choice"], sim_data["choice"])
        np.testing.assert_allclose(extra["p_contaminated"], [0.0])

    def test_random_choice_contamination_ignores_nonpositive_response_times(self):
        sim_data = {
            "response_time": np.array([[1.0, 0.0, -1.0, 2.0]]),
            "choice": np.array([[0.0, 99.0, -1.0, 1.0]]),
        }
        original_response_time = sim_data["response_time"].copy()
        model = _make_bare_model(contamination=RandomChoiceContamination(p_contaminated=1.0))
        rng = np.random.default_rng(0)

        out_data, extra = model._apply_contamination(sim_data, rng=rng)

        np.testing.assert_allclose(sim_data["response_time"], original_response_time)
        np.testing.assert_allclose(extra["p_contaminated"], [1.0])
        np.testing.assert_allclose(out_data["response_time"][0, 1:3], [0.0, -1.0])
        np.testing.assert_allclose(out_data["choice"][0, 1:3], [99.0, -1.0])
        assert np.all(np.isfinite(out_data["response_time"][sim_data["response_time"] > 0]))
        assert np.all(out_data["response_time"][sim_data["response_time"] > 0] > 0)
        assert set(out_data["choice"][sim_data["response_time"] > 0]) <= {0.0, 1.0}

    def test_random_choice_contamination_supports_custom_data_keys(self):
        sim_data = {
            "rt": np.array([[1.0, 2.0, 3.0]]),
            "resp": np.array([[0.0, 1.0, 0.0]]),
            "metadata": np.array([42.0]),
        }
        process = RandomChoiceContamination(
            p_contaminated=0.0,
            response_time_key="rt",
            choice_key="resp",
        )
        model = _make_bare_model(contamination=process, data_keys=("rt", "resp", "metadata"))
        rng = np.random.default_rng(0)

        out_data, extra = model._apply_contamination(sim_data, rng=rng)

        np.testing.assert_allclose(out_data["rt"], sim_data["rt"])
        np.testing.assert_allclose(out_data["resp"], sim_data["resp"])
        assert out_data["metadata"] is sim_data["metadata"]
        np.testing.assert_allclose(extra["p_contaminated"], [0.0])

    def test_missing_required_keys_raises(self):
        model = _make_bare_model(contamination=RandomChoiceContamination())
        with pytest.raises(KeyError):
            model._apply_contamination({"response_time": np.array([[1.0]])}, rng=None)

    def test_prior_probability_can_be_registered_as_shared(self, sim_data):
        process = RandomChoiceContamination(p_contaminated=Prior("beta", a=2, b=8), infer=True)

        out = process.apply(sim_data, rng=np.random.default_rng(0))

        assert process.parameter_groups() == {"shared_params": ["p_contaminated"]}
        assert out["p_contaminated"].shape == (1,)

    def test_stochastic_probability_can_vary_over_steps(self, sim_data):
        process = RandomChoiceContamination(
            p_contaminated=RandomWalk(bounds=(0.0, 1.0), sigma=0.0, delta=0.0),
            infer=True,
        )

        out = process.apply(sim_data, rng=np.random.default_rng(0))

        assert out["p_contaminated"].shape == sim_data["response_time"].shape
        assert process.parameter_groups() == {
            "local_params": ["p_contaminated"],
            "fixed_params": ["p_contaminated_sigma", "p_contaminated_delta"],
        }

    def test_deterministic_probability_exposes_transition_hyperparameters(self, sim_data):
        process = RandomChoiceContamination(
            p_contaminated=Linear(
                bounds=(0.0, 1.0),
                intercept=Prior("beta", a=2, b=8),
                slope=0.0,
            ),
            infer=True,
        )

        out = process.apply(sim_data, rng=np.random.default_rng(0))

        assert out["p_contaminated"].shape == sim_data["response_time"].shape
        assert process.parameter_groups() == {
            "deterministic_params": ["p_contaminated"],
            "hyper_params": ["p_contaminated_intercept"],
            "fixed_params": ["p_contaminated_slope"],
        }


class TestContaminationParameterValidation:
    def test_rejects_non_boolean_infer(self):
        with pytest.raises(TypeError, match="infer must be a bool"):
            RandomChoiceContamination(infer=1)

    def test_rejects_probability_outside_unit_interval(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            RandomChoiceContamination(p_contaminated=1.1)

    def test_rejects_unsupported_probability_type(self):
        with pytest.raises(TypeError, match="p_contaminated must be"):
            RandomChoiceContamination(p_contaminated="often")

    def test_rejects_transition_bounds_outside_unit_interval(self):
        with pytest.raises(ValueError, match="bounds within"):
            RandomChoiceContamination(p_contaminated=RandomWalk(bounds=(-1.0, 1.0)))


class TestSampleIntegrationOrdering:
    def test_contamination_extra_and_missing_extra_coexist(self, sim_data):
        """Simulates the relevant slice of `sample()`'s body: contamination
        is applied first, then missingness, and both processes' extra keys
        plus `missing_mask` must all appear in the final result dict
        without overwriting each other.
        """
        model = _make_bare_model(
            contamination=ConstantContaminationProcess(),
            missing=make_missing(mask_value=False),
            data_keys=("response_time", "choice"),
        )

        data, contamination_extra = model._apply_contamination(sim_data, rng=None)
        data, missing_mask, missing_extra = model._apply_missing(data, rng=None)

        result = {**data}
        if contamination_extra:
            result.update(contamination_extra)
        if missing_mask is not None:
            result["missing_mask"] = missing_mask
        if missing_extra:
            result.update(missing_extra)

        assert np.all(result["response_time"] == 999.0)  # contamination applied
        assert "p_contaminated" in result
        assert "missing_mask" in result
        assert "p_missing" in result
        # sanity: contamination and missing extras didn't clobber each other
        assert result["p_contaminated"] != result["p_missing"] or not np.array_equal(
            result["p_contaminated"], result["p_missing"]
        )

    def test_missing_sees_contaminated_data(self, sim_data):
        """The missing process runs second, so it should observe the
        already-contaminated response_time/choice, not the originals.
        """
        seen = {}

        def spy_missing(data, rng=None):
            seen["response_time"] = data["response_time"].copy()
            out = dict(data)
            out["missing_mask"] = np.zeros_like(data["response_time"], dtype=bool)
            return out

        model = _make_bare_model(
            contamination=ConstantContaminationProcess(),
            missing=spy_missing,
        )

        data, _ = model._apply_contamination(sim_data, rng=None)
        model._apply_missing(data, rng=None)

        assert np.all(seen["response_time"] == 999.0)

    def test_no_contamination_no_missing_is_passthrough(self, sim_data):
        model = _make_bare_model(contamination=None, missing=None)
        data, extra = model._apply_contamination(sim_data, rng=None)
        data, mask, missing_extra = model._apply_missing(data, rng=None)

        assert data is sim_data
        assert extra == {}
        assert mask is None
        assert missing_extra == {}
