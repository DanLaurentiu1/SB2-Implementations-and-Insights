import pytest
import numpy as np
from environments.custom_envs.StationaryKArmEnvironment import StationaryKArmEnvironment


# GIVEN
@pytest.fixture
def primary_env_seed_16() -> StationaryKArmEnvironment:
    return StationaryKArmEnvironment(number_of_arms=2, seed=16, max_steps=10)


@pytest.fixture
def secondary_env_seed_16() -> StationaryKArmEnvironment:
    return StationaryKArmEnvironment(number_of_arms=2, seed=16, max_steps=10)


@pytest.fixture
def primary_env_seed_17() -> StationaryKArmEnvironment:
    return StationaryKArmEnvironment(number_of_arms=2, seed=17, max_steps=10)


@pytest.fixture
def secondary_env_seed_17() -> StationaryKArmEnvironment:
    return StationaryKArmEnvironment(number_of_arms=2, seed=17, max_steps=10)


def test_get_new_arms_match_when_same_seed(
    primary_env_seed_16: StationaryKArmEnvironment,
    secondary_env_seed_16: StationaryKArmEnvironment,
):
    # WHEN
    primary_env_seed_16._get_new_arms()
    secondary_env_seed_16._get_new_arms()

    # THEN
    assert (
        np.array_equal(primary_env_seed_16.arm_means, secondary_env_seed_16.arm_means)
        == True
    )


def test_get_new_arms_do_not_match_when_different_seed(
    primary_env_seed_16: StationaryKArmEnvironment,
    primary_env_seed_17: StationaryKArmEnvironment,
):
    # WHEN
    primary_env_seed_16._get_new_arms()
    primary_env_seed_17._get_new_arms()

    # THEN
    assert (
        np.array_equal(primary_env_seed_16.arm_means, primary_env_seed_17.arm_means)
        == False
    )


def test_get_new_arms_reward_matches_when_same_seed(
    primary_env_seed_16: StationaryKArmEnvironment,
    secondary_env_seed_16: StationaryKArmEnvironment,
):
    # WHEN
    _, primary_reward, _, _, _ = primary_env_seed_16.step(0)
    _, secondary_reward, _, _, _ = secondary_env_seed_16.step(0)

    # THEN
    assert np.equal(primary_reward, secondary_reward) == True


def test_reset_none_does_not_change_internal_seeding(
    primary_env_seed_16: StationaryKArmEnvironment,
):
    # WHEN
    old_arm_means = primary_env_seed_16.arm_means
    primary_env_seed_16.reset()
    new_arm_means = primary_env_seed_16.arm_means

    # THEN
    assert primary_env_seed_16.seed == 16
    assert np.array_equal(old_arm_means, new_arm_means) == True


def test_reset_new_does_change_internal_seeding(
    primary_env_seed_16: StationaryKArmEnvironment,
    primary_env_seed_17: StationaryKArmEnvironment,
):
    # WHEN
    primary_env_seed_16.reset(seed=17)
    reset_seed_16_arm_means = primary_env_seed_16.arm_means
    seed_17_arm_means = primary_env_seed_17.arm_means

    # THEN
    assert np.array_equal(reset_seed_16_arm_means, seed_17_arm_means) == True
