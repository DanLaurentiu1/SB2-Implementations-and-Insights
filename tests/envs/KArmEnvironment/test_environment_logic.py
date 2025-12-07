from functools import partial
import pytest
import numpy as np
from environments.custom_envs.BanditEnvs.KArmEnvironment import (
    KArmEnvironment,
)
from environments.custom_envs.BanditEnvs.drift.GaussianDrift import GaussianDrift
from gymnasium import Space
from gymnasium.spaces import Discrete
from environments.custom_envs.BanditEnvs.reward.GaussianReward import GaussianReward
from utils.exceptions.logic_exceptions import EnvironmentLogicException


# GIVEN
@pytest.fixture
def stationary_env() -> KArmEnvironment:
    return KArmEnvironment(number_of_arms=2, seed=16, max_steps=10)


# GIVEN
@pytest.fixture
def non_stationary_env() -> KArmEnvironment:
    return KArmEnvironment(
        number_of_arms=10,
        seed=16,
        max_steps=100,
        drift_factory=partial(GaussianDrift, mean=0, variance=0.01),
    )


def test_environment_initialization(
    stationary_env: KArmEnvironment,
):
    # WHEN
    number_of_arms = 2
    seed = 16
    max_steps = 10
    observation_space: Space = Discrete(1, seed=seed)

    # THEN
    assert stationary_env.number_of_arms == number_of_arms
    assert stationary_env.seed == seed
    assert stationary_env.max_steps == max_steps
    assert stationary_env.observation_space == observation_space


def test_get_info(stationary_env: KArmEnvironment):
    # WHEN
    optimal_arm_false = False
    optimal_arm_true = True

    # THEN
    assert isinstance(
        stationary_env._get_info(optimal_arm_chosen=optimal_arm_false), dict
    )
    assert isinstance(
        stationary_env._get_info(optimal_arm_chosen=optimal_arm_true), dict
    )


def test_get_obs(stationary_env: KArmEnvironment):
    # THEN
    assert isinstance(stationary_env._get_obs(), np.float64)


def test_get_new_arms(stationary_env: KArmEnvironment):
    # WHEN
    stationary_env._get_new_arms()
    optimal_arm_index = int(np.argmax(stationary_env._arm_means))

    # THEN
    assert isinstance(stationary_env.arm_means, np.ndarray)
    assert isinstance(stationary_env.arm_means[0], np.float64)
    assert len(stationary_env.arm_means) == stationary_env.number_of_arms
    assert optimal_arm_index == stationary_env.optimal_arm
    assert 0 <= optimal_arm_index < stationary_env.number_of_arms


def test_reset(stationary_env: KArmEnvironment):
    # WHEN
    previous_arms = stationary_env.arm_means
    reset_observation, reset_information = stationary_env.reset()
    current_arms = stationary_env.arm_means

    # THEN
    assert stationary_env._pulls == 0
    assert np.array_equal(previous_arms, current_arms) == True
    assert isinstance(reset_information, dict)
    assert isinstance(reset_observation, np.float64)


def test_step_output(stationary_env: KArmEnvironment):
    # WHEN
    step_obs, step_reward, step_terminated, step_truncated, step_info = (
        stationary_env.step(0)
    )

    # THEN
    assert isinstance(step_obs, np.float64)
    assert isinstance(step_reward, float)
    assert isinstance(step_truncated, bool)
    assert isinstance(step_terminated, bool)
    assert isinstance(step_info, dict)


def test_step_terminated(stationary_env: KArmEnvironment):
    # WHEN
    for _ in range(1, stationary_env._max_steps):
        assert stationary_env._terminated == False
        stationary_env.step(0)
    _, _, step_terminated, _, _ = stationary_env.step(0)

    # THEN
    assert step_terminated == True


def test_step_reward_values(stationary_env: KArmEnvironment):
    # WHEN
    for index, mean in enumerate(stationary_env.arm_means):
        _, reward, _, _, _ = stationary_env.step(index)

        # THEN
        assert mean - 3 <= reward <= mean + 3


def test_ensure_not_terminated_correctly_called(
    stationary_env: KArmEnvironment,
):
    # WHEN
    for _ in range(stationary_env.max_steps):
        stationary_env.step(0)

    # THEN
    with pytest.raises(EnvironmentLogicException) as exception_output:
        stationary_env.step(0)
    assert "step() called after rollout termination. call reset() first." in str(
        exception_output.value
    )


def test_validate_input_number_of_arms_invalid():
    # WHEN
    with pytest.raises(EnvironmentLogicException) as exception_output:
        KArmEnvironment(number_of_arms=-1, seed=16, max_steps=10)

    # THEN
    assert (
        "Invalid number of arms=-1. This number must be positive and bigger than 0."
        in str(exception_output.value)
    )


def test_validate_input_seed_invalid():
    # WHEN
    with pytest.raises(EnvironmentLogicException) as exception_output:
        KArmEnvironment(number_of_arms=2, seed=-1, max_steps=10)

    # THEN
    assert "Invalid seed=-1. This number must be positive." in str(
        exception_output.value
    )


def test_validate_input_max_steps_invalid():
    # WHEN
    with pytest.raises(EnvironmentLogicException) as exception_output:
        KArmEnvironment(number_of_arms=12, seed=16, max_steps=-1)

    # THEN
    assert (
        "Invalid number of max_steps=-1. This number must be positive and bigger than 0."
        in str(exception_output.value)
    )


def test_validate_action_invalid(stationary_env: KArmEnvironment):
    # WHEN
    with pytest.raises(EnvironmentLogicException) as exception_output:
        stationary_env.step(action=3)

    # THEN
    assert "Invalid action=3. Action must be a member of [0, 1]" in str(
        exception_output.value
    )


def test_step_non_stationary_optimal_arm_stays_consistent(
    non_stationary_env: KArmEnvironment,
):
    # WHEN
    non_stationary_env.reset()
    terminated = truncated = False

    while not terminated and not truncated:

        # THEN
        assert non_stationary_env.optimal_arm == np.argmax(non_stationary_env.arm_means)
        _, _, terminated, truncated, _ = non_stationary_env.step(action=0)


def test_environment_to_string_method(stationary_env: KArmEnvironment):
    # WHEN
    expected_string = "KArmEnvironment(seed=16, arms=2)"
    actual_string = stationary_env.__str__()

    # THEN
    assert expected_string == actual_string


def test_environment_to_repr_method(stationary_env: KArmEnvironment):
    # WHEN
    expected_string = "KArmEnvironment(\n\tdrift=NoDrift(),\n\treward=GaussianReward(variance=1),\n\tseed=16,\n\tarms=2\n)"
    actual_string = stationary_env.__repr__()

    # THEN
    assert expected_string == actual_string
