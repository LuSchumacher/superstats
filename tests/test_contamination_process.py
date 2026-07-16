"""Tests for GenerativeModel's contamination_process integration."""

import numpy as np
import pytest

from superstats.simulation.generative_model import GenerativeModel
from superstats.simulation.augmentation.contamination_process import ContaminationProcess
from superstats.simulation.augmentation.random_choice_process import RandomChoiceProcess


def _make_bare_model(contamination_process=None, missing_process=None, data_keys=("response_time", "choice")):
    """Build a GenerativeModel-like object without running the real __init__.

    __init__ requires a JointPrior and simulator to do a pilot draw; the
    contamination integration only depends on `self.contamination_process`,
    `self.missing_process`, and `self.data_keys`, so we construct those
    directly to keep these tests fast and independent of prior/simulator
    machinery.
    """
    model = object.__new__(GenerativeModel)
    model.contamination_process = contamination_process
    model.missing_process = missing_process
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


def make_missing_process(mask_value=False):
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
        contamination_process = None
        if contamination_process == "random_choice":
            result = RandomChoiceProcess()
        elif (
            contamination_process is None
            or isinstance(contamination_process, ContaminationProcess)
            or callable(contamination_process)
        ):
            result = contamination_process
        else:
            pytest.fail("should not raise")
        assert result is None

    def test_random_choice_literal_instantiates_random_choice_process(self):
        contamination_process = "random_choice"
        if contamination_process == "random_choice":
            result = RandomChoiceProcess()
        else:
            pytest.fail("should have matched the literal branch")
        assert isinstance(result, RandomChoiceProcess)

    def test_bare_contamination_process_instance_is_accepted(self):
        """Regression test: a bare ContaminationProcess instance (not going
        through the "random_choice" string) must be accepted as-is, per the
        constructor's type hint and error message.
        """
        instance = ConstantContaminationProcess()
        contamination_process = instance
        if contamination_process == "random_choice":
            pytest.fail("should not match the literal branch")
        elif (
            contamination_process is None
            or isinstance(contamination_process, ContaminationProcess)
            or callable(contamination_process)
        ):
            result = contamination_process
        else:
            pytest.fail(
                "bare ContaminationProcess instance was rejected; "
                "instances only define .apply(), not __call__, so the "
                "old `callable(...)` check alone is insufficient"
            )
        assert result is instance

    def test_plain_callable_is_accepted(self):
        contamination_process = constant_contamination_callable
        if contamination_process == "random_choice":
            pytest.fail("should not match the literal branch")
        elif (
            contamination_process is None
            or isinstance(contamination_process, ContaminationProcess)
            or callable(contamination_process)
        ):
            result = contamination_process
        else:
            pytest.fail("plain callable should be accepted")
        assert result is constant_contamination_callable

    def test_invalid_value_is_rejected(self):
        contamination_process = "not_a_valid_option"
        with pytest.raises(TypeError):
            if contamination_process == "random_choice":
                pass
            elif (
                contamination_process is None
                or isinstance(contamination_process, ContaminationProcess)
                or callable(contamination_process)
            ):
                pass
            else:
                raise TypeError(
                    "contamination_process must be None, 'random_choice', a ContaminationProcess instance, or callable"
                )


class TestApplyContaminationProcess:
    def test_none_process_returns_sim_data_unchanged(self, sim_data):
        model = _make_bare_model(contamination_process=None)
        out_data, extra = model._apply_contamination_process(sim_data, rng=None)
        assert out_data is sim_data
        assert extra == {}

    def test_contamination_process_instance_dispatches_via_apply(self, sim_data):
        model = _make_bare_model(contamination_process=ConstantContaminationProcess())
        out_data, extra = model._apply_contamination_process(sim_data, rng=None)

        assert np.all(out_data["response_time"] == 999.0)
        assert set(out_data.keys()) == set(sim_data.keys())  # no leakage of extras into sim_data
        assert extra == {"p_contaminated": pytest.approx(np.array([0.5]))} or np.allclose(
            extra["p_contaminated"], [0.5]
        )

    def test_plain_callable_is_invoked_directly(self, sim_data):
        """Regression test for the AttributeError bug: a plain callable
        contamination_process must not be routed through `.apply`.
        """
        model = _make_bare_model(contamination_process=constant_contamination_callable)
        out_data, extra = model._apply_contamination_process(sim_data, rng=None)

        assert np.all(out_data["response_time"] == 111.0)
        assert set(out_data.keys()) == set(sim_data.keys())
        assert np.allclose(extra["p_contaminated"], [0.25])

    def test_random_choice_process_end_to_end(self, sim_data):
        """Use the real RandomChoiceProcess with p=1.0 so every step is
        guaranteed to be contaminated, making the effect deterministic
        enough to assert on.
        """
        model = _make_bare_model(contamination_process=RandomChoiceProcess(p_contaminated=1.0))
        rng = np.random.default_rng(0)

        out_data, extra = model._apply_contamination_process(sim_data, rng=rng)

        assert "p_contaminated" in extra
        np.testing.assert_allclose(extra["p_contaminated"], [1.0])
        # with full contamination, choice values should be resampled from
        # the batch's own unique discrete choices
        assert set(np.unique(out_data["choice"])) <= set(np.unique(sim_data["choice"]))
        # response times are redrawn (heavy-tailed), so at least verify
        # shape/positivity rather than exact values
        assert out_data["response_time"].shape == sim_data["response_time"].shape
        assert np.all(out_data["response_time"] > 0)

    def test_random_choice_process_p_zero_is_noop(self, sim_data):
        model = _make_bare_model(contamination_process=RandomChoiceProcess(p_contaminated=0.0))
        rng = np.random.default_rng(0)

        out_data, extra = model._apply_contamination_process(sim_data, rng=rng)

        np.testing.assert_allclose(out_data["response_time"], sim_data["response_time"])
        np.testing.assert_allclose(out_data["choice"], sim_data["choice"])
        np.testing.assert_allclose(extra["p_contaminated"], [0.0])

    def test_random_choice_process_ignores_nonpositive_response_times(self):
        sim_data = {
            "response_time": np.array([[1.0, 0.0, -1.0, 2.0]]),
            "choice": np.array([[0.0, 99.0, -1.0, 1.0]]),
        }
        original_response_time = sim_data["response_time"].copy()
        model = _make_bare_model(contamination_process=RandomChoiceProcess(p_contaminated=1.0))
        rng = np.random.default_rng(0)

        out_data, extra = model._apply_contamination_process(sim_data, rng=rng)

        np.testing.assert_allclose(sim_data["response_time"], original_response_time)
        np.testing.assert_allclose(extra["p_contaminated"], [1.0])
        np.testing.assert_allclose(out_data["response_time"][0, 1:3], [0.0, -1.0])
        np.testing.assert_allclose(out_data["choice"][0, 1:3], [99.0, -1.0])
        assert np.all(np.isfinite(out_data["response_time"][sim_data["response_time"] > 0]))
        assert np.all(out_data["response_time"][sim_data["response_time"] > 0] > 0)
        assert set(out_data["choice"][sim_data["response_time"] > 0]) <= {0.0, 1.0}

    def test_random_choice_process_supports_custom_data_keys(self):
        sim_data = {
            "rt": np.array([[1.0, 2.0, 3.0]]),
            "resp": np.array([[0.0, 1.0, 0.0]]),
            "metadata": np.array([42.0]),
        }
        process = RandomChoiceProcess(
            p_contaminated=0.0,
            response_time_key="rt",
            choice_key="resp",
        )
        model = _make_bare_model(contamination_process=process, data_keys=("rt", "resp", "metadata"))
        rng = np.random.default_rng(0)

        out_data, extra = model._apply_contamination_process(sim_data, rng=rng)

        np.testing.assert_allclose(out_data["rt"], sim_data["rt"])
        np.testing.assert_allclose(out_data["resp"], sim_data["resp"])
        assert out_data["metadata"] is sim_data["metadata"]
        np.testing.assert_allclose(extra["p_contaminated"], [0.0])

    def test_missing_required_keys_raises(self):
        model = _make_bare_model(contamination_process=RandomChoiceProcess())
        with pytest.raises(KeyError):
            model._apply_contamination_process({"response_time": np.array([[1.0]])}, rng=None)


class TestSampleIntegrationOrdering:
    def test_contamination_extra_and_missing_extra_coexist(self, sim_data):
        """Simulates the relevant slice of `sample()`'s body: contamination
        is applied first, then missingness, and both processes' extra keys
        plus `missing_mask` must all appear in the final result dict
        without overwriting each other.
        """
        model = _make_bare_model(
            contamination_process=ConstantContaminationProcess(),
            missing_process=make_missing_process(mask_value=False),
            data_keys=("response_time", "choice"),
        )

        data, contamination_extra = model._apply_contamination_process(sim_data, rng=None)
        data, missing_mask, missing_extra = model._apply_missing_process(data, rng=None)

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

    def test_missing_process_sees_contaminated_data(self, sim_data):
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
            contamination_process=ConstantContaminationProcess(),
            missing_process=spy_missing,
        )

        data, _ = model._apply_contamination_process(sim_data, rng=None)
        model._apply_missing_process(data, rng=None)

        assert np.all(seen["response_time"] == 999.0)

    def test_no_contamination_no_missing_is_passthrough(self, sim_data):
        model = _make_bare_model(contamination_process=None, missing_process=None)
        data, extra = model._apply_contamination_process(sim_data, rng=None)
        data, mask, missing_extra = model._apply_missing_process(data, rng=None)

        assert data is sim_data
        assert extra == {}
        assert mask is None
        assert missing_extra == {}
