import pytest
import numpy as np
from environments.custom_envs.BanditEnvs.KArmEnvironment import (
    KArmEnvironment,
)
from environments.custom_envs.BanditEnvs.drift.GaussianDrift import GaussianDrift
from environments.custom_envs.BanditEnvs.drift.NoDrift import NoDrift
from utils.exceptions.logic_exceptions import DriftingLogicException
from gymnasium.utils.seeding import np_random
from contextlib import contextmanager


@contextmanager
def not_raises():
    try:
        yield
    except Exception as e:
        raise AssertionError(f"Raised unexpected exception: {e}")


# GIVEN
@pytest.fixture
def no_drift() -> NoDrift:
    return NoDrift()


# GIVEN
@pytest.fixture
def gaussian_drift() -> GaussianDrift:
    return GaussianDrift()


# GIVEN
@pytest.fixture
def first_rng_16() -> np.random.Generator:
    rng, _ = np_random(seed=16)
    return rng


# GIVEN
@pytest.fixture
def second_rng_16() -> np.random.Generator:
    rng, _ = np_random(seed=16)
    return rng


# GIVEN
@pytest.fixture
def arm_means() -> np.ndarray:
    rng, _ = np_random(seed=16)
    return rng.normal(loc=0.0, scale=1.0, size=5)


def test_no_drift_can_call_all_methods(
    no_drift: NoDrift, arm_means: np.ndarray, first_rng_16: np.random.Generator
):
    # THEN
    with not_raises():
        no_drift.drift(arm_means=arm_means, rng=first_rng_16)
        no_drift.reset()


def test_no_drift_does_not_change_arm_means(
    no_drift: NoDrift, arm_means: np.ndarray, first_rng_16: np.random.Generator
):
    # WHEN
    original_arm_means = arm_means
    new_arm_means = no_drift.drift(arm_means=arm_means, rng=first_rng_16)

    # THEN
    assert np.array_equal(original_arm_means, new_arm_means) == True


def test_gaussian_drift_can_call_all_methods(
    gaussian_drift: GaussianDrift,
    arm_means: np.ndarray,
    first_rng_16: np.random.Generator,
):
    # THEN
    with not_raises():
        gaussian_drift.drift(arm_means=arm_means, rng=first_rng_16)
        gaussian_drift.reset()


def test_gaussian_drift_negative_variance_throws_exception():
    # WHEN
    with pytest.raises(DriftingLogicException) as exception_output:
        GaussianDrift(mean=1.0, variance=-1)

    # THEN
    assert "Invalid variance=-1. Variance cannot be negative." in str(
        exception_output.value
    )


def test_gaussian_drift_out_of_bounds_variance_throws_exception():
    # WHEN
    with pytest.raises(DriftingLogicException) as exception_output:
        GaussianDrift(mean=1.0, variance=1e309)

    # THEN
    assert "Invalid variance=inf. Variance must be finite and within ±" in str(
        exception_output.value
    )


def test_gaussian_drift_out_of_bounds_mean_throws_exception():
    # WHEN
    with pytest.raises(DriftingLogicException) as exception_output:
        GaussianDrift(mean=1e390, variance=1.0)

    # THEN
    assert "Invalid mean=inf. Mean must be finite and within ±" in str(
        exception_output.value
    )


def test_gaussian_drift_drift_performs_correct_addition(
    gaussian_drift: GaussianDrift,
    arm_means: np.ndarray,
    first_rng_16: np.random.Generator,
    second_rng_16: np.random.Generator,
):
    # WHEN
    mean, variance = gaussian_drift.mean, gaussian_drift.variance
    expected_walk = first_rng_16.normal(loc=mean, scale=variance, size=arm_means.size)
    expected_arm_means = expected_walk + arm_means
    actual_arm_means = gaussian_drift.drift(arm_means=arm_means, rng=second_rng_16)

    # THEN
    assert np.array_equal(expected_arm_means, actual_arm_means) == True


def test_gaussian_drift_drift_is_deterministic(
    gaussian_drift: GaussianDrift,
    arm_means: np.ndarray,
    first_rng_16: np.random.Generator,
    second_rng_16: np.random.Generator,
):
    # WHEN
    mean, variance = gaussian_drift.mean, gaussian_drift.variance
    expected_arm_means_temp, actual_arm_means_temp = arm_means, arm_means
    for _ in range(100):
        expected_walk = first_rng_16.normal(
            loc=mean, scale=variance, size=expected_arm_means_temp.size
        )
        expected_arm_means_temp = expected_walk + expected_arm_means_temp
        actual_arm_means_temp = gaussian_drift.drift(
            arm_means=actual_arm_means_temp, rng=second_rng_16
        )

        # THEN
        assert np.array_equal(expected_arm_means_temp, actual_arm_means_temp) == True


def test_gaussian_drift_representation(gaussian_drift: GaussianDrift):
    # WHEN
    expected_string = "GaussianDrift(mean=0, variance=0.01)"
    actual_string = gaussian_drift.__repr__()
    # THEN
    assert expected_string == actual_string
