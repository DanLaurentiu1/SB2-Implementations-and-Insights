import pytest
import numpy as np
from environments.custom_envs.KArmEnvironment import KArmEnvironment


# GIVEN
@pytest.fixture
def primary_env_seed_16() -> KArmEnvironment:
    return KArmEnvironment(number_of_arms=2, seed=16, max_steps=10)


@pytest.fixture
def secondary_env_seed_16() -> KArmEnvironment:
    return KArmEnvironment(number_of_arms=2, seed=16, max_steps=10)


@pytest.fixture
def primary_env_seed_17() -> KArmEnvironment:
    return KArmEnvironment(number_of_arms=2, seed=17, max_steps=10)


@pytest.fixture
def secondary_env_seed_17() -> KArmEnvironment:
    return KArmEnvironment(number_of_arms=2, seed=17, max_steps=10)


def test_get_new_arms_match_when_same_seed(
    primary_env_seed_16: KArmEnvironment, secondary_env_seed_16: KArmEnvironment
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
    primary_env_seed_16: KArmEnvironment, primary_env_seed_17: KArmEnvironment
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
    primary_env_seed_16: KArmEnvironment, secondary_env_seed_16: KArmEnvironment
):
    # WHEN
    _, primary_reward, _, _, _ = primary_env_seed_16.step(0)
    _, secondary_reward, _, _, _ = secondary_env_seed_16.step(0)

    # THEN
    assert np.equal(primary_reward, secondary_reward) == True
