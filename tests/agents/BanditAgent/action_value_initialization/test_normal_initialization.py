import numpy as np
import pytest

from algorithms.bandits.implementations.action_value_initialization.NormalActionValueInitialization import (
    NormalActionValueInitialization,
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
def normal_action_value_initialization_strategy(
    test_env: BaseBanditEnv,
) -> NormalActionValueInitialization:
    strategy = NormalActionValueInitialization()
    strategy._setup(env=test_env)
    return strategy


# ==============
# Tests
# ==============


def test_normal_initialization_construction_values(
    normal_action_value_initialization_strategy: NormalActionValueInitialization,
    test_env: KArmEnvironment,
):
    # THEN
    assert (
        normal_action_value_initialization_strategy._number_of_arms
        == test_env.number_of_arms
    )


def test_normal_initialization_has_expected_values(
    normal_action_value_initialization_strategy: NormalActionValueInitialization,
    test_env: KArmEnvironment,
):
    # WHEN
    expected_q_values = np.zeros(shape=test_env.number_of_arms)
    actual_q_values = normal_action_value_initialization_strategy.init_action_values()

    # THEN
    assert np.array_equal(expected_q_values, actual_q_values)


def test_normal_initialization_repr(
    normal_action_value_initialization_strategy: NormalActionValueInitialization,
):
    # WHEN
    expected_string = "NormalAVInit"
    actual_string = normal_action_value_initialization_strategy.__repr__()

    # THEN
    assert expected_string == actual_string
