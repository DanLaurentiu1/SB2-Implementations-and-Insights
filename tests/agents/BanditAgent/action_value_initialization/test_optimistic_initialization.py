import numpy as np
import pytest

from algorithms.bandits.implementations.action_value_initialization.OptimisticActionValueInitialization import (
    OptimisticActionValueInitialization,
)
from environments.custom_envs.BanditEnvs.KArmEnvironment import KArmEnvironment
from environments.custom_envs.BanditEnvs.BaseBanditEnv import BaseBanditEnv

# ==============
# Fixtures
# ==============


# GIVEN
@pytest.fixture
def test_env() -> BaseBanditEnv:
    return KArmEnvironment(number_of_arms=3, seed=16, max_steps=10)


# GIVEN
@pytest.fixture
def optimistic_action_value_initialization_strategy(
    test_env: BaseBanditEnv,
) -> OptimisticActionValueInitialization:
    strategy = OptimisticActionValueInitialization()
    strategy._setup(env=test_env)
    return strategy


# ==============
# Tests
# ==============


def test_optimistic_initialization_construction_values(
    optimistic_action_value_initialization_strategy: OptimisticActionValueInitialization,
    test_env: KArmEnvironment,
):
    # THEN
    assert (
        optimistic_action_value_initialization_strategy._number_of_arms
        == test_env.number_of_arms
    )

    assert (
        optimistic_action_value_initialization_strategy._optimal_arm_mean_value
        == test_env.arms[test_env.optimal_arm]
    )

    assert (
        optimistic_action_value_initialization_strategy._env_reward_variance
        == test_env.reward_strategy.variance
    )


def test_optimistic_initialization_has_expected_values(
    optimistic_action_value_initialization_strategy: OptimisticActionValueInitialization,
    test_env: KArmEnvironment,
):
    # WHEN
    variance = test_env.reward_strategy.variance
    optimal_arm_mean = test_env.arms[test_env.optimal_arm]
    expected_q_values = np.full(
        shape=test_env.number_of_arms, fill_value=optimal_arm_mean + 4 * variance
    )
    actual_q_values = (
        optimistic_action_value_initialization_strategy.init_action_values()
    )

    # THEN
    assert np.array_equal(expected_q_values, actual_q_values)


def test_optimistic_initialization_repr(
    optimistic_action_value_initialization_strategy: OptimisticActionValueInitialization,
):
    # WHEN
    expected_string = "OptimisticAVInit"
    actual_string = optimistic_action_value_initialization_strategy.__repr__()

    # THEN
    assert expected_string == actual_string
