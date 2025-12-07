from contextlib import contextmanager

import numpy as np
import pytest
from gymnasium.utils.seeding import np_random

from environments.custom_envs.BanditEnvs.reward.GaussianReward import GaussianReward
from utils.exceptions.logic_exceptions import RewardLogicException


@contextmanager
def not_raises():
    try:
        yield
    except Exception as e:
        raise AssertionError(f"Raised unexpected exception: {e}")


# GIVEN
@pytest.fixture
def gaussian_reward() -> GaussianReward:
    return GaussianReward(variance=1)


# GIVEN
@pytest.fixture
def arm_means() -> np.ndarray:
    rng, _ = np_random(seed=16)
    return rng.normal(loc=0.0, scale=1.0, size=5)


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


def test_gaussian_reward_can_call_methods(
    gaussian_reward: GaussianReward,
    arm_means: np.ndarray,
    first_rng_16: np.random.Generator,
):
    # THEN
    with not_raises():
        gaussian_reward.get_reward(arm_mean=arm_means[0], rng=first_rng_16)


def test_gaussian_reward_negative_variance_throws_exception():
    # WHEN
    with pytest.raises(RewardLogicException) as exception_output:
        GaussianReward(variance=-1.0)

    # THEN
    assert "Invalid variance=-1.0. Variance cannot be negative." in str(
        exception_output.value
    )


def test_gaussian_reward_infinite_variance_throws_exception():
    # WHEN
    with pytest.raises(RewardLogicException) as exception_output:
        GaussianReward(variance=1e309)

    # THEN
    assert "Invalid variance=inf. Variance must be finite and within ±" in str(
        exception_output.value
    )


def test_gaussian_reward_get_reward_is_deterministic(
    gaussian_reward: GaussianReward,
    arm_means: np.ndarray,
    first_rng_16: np.random.Generator,
    second_rng_16: np.random.Generator,
):
    # WHEN
    arm_mean = arm_means[0]
    for _ in range(100):
        expected_reward = second_rng_16.normal(loc=arm_mean, scale=1.0)
        actual_reward = gaussian_reward.get_reward(arm_mean=arm_mean, rng=first_rng_16)

        # THEN
        assert np.equal(expected_reward, actual_reward)
