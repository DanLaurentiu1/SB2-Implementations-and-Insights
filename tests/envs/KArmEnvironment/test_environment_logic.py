import pytest
import numpy as np
from environments.custom_envs.StationaryKArmEnvironment import StationaryKArmEnvironment
from utils.exceptions.logic_exceptions import EnvironmentLogicException


# GIVEN
@pytest.fixture
def simple_bandit_environment() -> StationaryKArmEnvironment:
    return StationaryKArmEnvironment(number_of_arms=2, seed=16, max_steps=10)


def test_environment_initialization(
    simple_bandit_environment: StationaryKArmEnvironment,
):
    # WHEN
    number_of_arms = 2
    seed = 16
    max_steps = 10

    # THEN
    assert simple_bandit_environment.number_of_arms == number_of_arms
    assert simple_bandit_environment.seed == seed
    assert simple_bandit_environment.max_steps == max_steps


def test_get_info(simple_bandit_environment: StationaryKArmEnvironment):
    # WHEN
    optimal_arm_false = False
    optimal_arm_true = True

    # THEN
    assert isinstance(
        simple_bandit_environment._get_info(optimal_arm_chosen=optimal_arm_false), dict
    )
    assert isinstance(
        simple_bandit_environment._get_info(optimal_arm_chosen=optimal_arm_true), dict
    )


def test_get_obs(simple_bandit_environment: StationaryKArmEnvironment):
    # THEN
    assert isinstance(simple_bandit_environment._get_obs(), np.float64)


def test_get_new_arms(simple_bandit_environment: StationaryKArmEnvironment):
    # WHEN
    simple_bandit_environment._get_new_arms()
    optimal_arm_index = int(np.argmax(simple_bandit_environment.arm_means))

    # THEN
    assert isinstance(simple_bandit_environment.arm_means, np.ndarray)
    assert isinstance(simple_bandit_environment.arm_means[0], np.float64)
    assert (
        len(simple_bandit_environment.arm_means)
        == simple_bandit_environment.number_of_arms
    )
    assert optimal_arm_index == simple_bandit_environment.optimal_arm
    assert 0 <= optimal_arm_index < simple_bandit_environment.number_of_arms


def test_reset(simple_bandit_environment: StationaryKArmEnvironment):
    # WHEN
    previous_arms = simple_bandit_environment.arm_means
    reset_observation, reset_information = simple_bandit_environment.reset()
    current_arms = simple_bandit_environment.arm_means

    # THEN
    assert simple_bandit_environment.pulls == 0
    assert np.array_equal(previous_arms, current_arms) == True
    assert isinstance(reset_information, dict)
    assert isinstance(reset_observation, np.float64)


def test_step_output(simple_bandit_environment: StationaryKArmEnvironment):
    # WHEN
    step_obs, step_reward, step_terminated, step_truncated, step_info = (
        simple_bandit_environment.step(0)
    )
    assert isinstance(step_obs, np.float64)
    assert isinstance(step_reward, float)
    assert isinstance(step_truncated, bool)
    assert isinstance(step_terminated, bool)
    assert isinstance(step_info, dict)


def test_step_terminated(simple_bandit_environment: StationaryKArmEnvironment):
    # WHEN
    for _ in range(1, simple_bandit_environment.max_steps):
        assert simple_bandit_environment.terminated == False
        simple_bandit_environment.step(0)
    _, _, step_terminated, _, _ = simple_bandit_environment.step(0)

    # THEN
    assert step_terminated == True


def test_step_reward_values(simple_bandit_environment: StationaryKArmEnvironment):
    # WHEN
    for index, mean in enumerate(simple_bandit_environment.arm_means):
        _, reward, _, _, _ = simple_bandit_environment.step(index)

        # THEN
        assert mean - 3 <= reward <= mean + 3


def test_ensure_not_terminated_correctly_called(
    simple_bandit_environment: StationaryKArmEnvironment,
):
    # WHEN
    for _ in range(simple_bandit_environment.max_steps):
        simple_bandit_environment.step(0)

    # THEN
    with pytest.raises(EnvironmentLogicException) as exception_output:
        simple_bandit_environment.step(0)
    assert "step() called after rollout termination. call reset() first." in str(
        exception_output.value
    )


def test_validate_input_number_of_arms_invalid():
    # WHEN
    with pytest.raises(EnvironmentLogicException) as exception_output:
        StationaryKArmEnvironment(number_of_arms=-1, seed=16, max_steps=10)

    # THEN
    assert (
        "Invalid number of arms=-1. This number must be positive and bigger than 0."
        in str(exception_output.value)
    )


def test_validate_input_seed_invalid():
    # WHEN
    with pytest.raises(EnvironmentLogicException) as exception_output:
        StationaryKArmEnvironment(number_of_arms=2, seed=-1, max_steps=10)

    # THEN
    assert "Invalid seed=-1. This number must be positive." in str(
        exception_output.value
    )


def test_validate_input_max_steps_invalid():
    # WHEN
    with pytest.raises(EnvironmentLogicException) as exception_output:
        StationaryKArmEnvironment(number_of_arms=12, seed=16, max_steps=-1)

    # THEN
    assert (
        "Invalid number of max_steps=-1. This number must be positive and bigger than 0."
        in str(exception_output.value)
    )


def test_validate_action_invalid(simple_bandit_environment: StationaryKArmEnvironment):
    # WHEN
    with pytest.raises(EnvironmentLogicException) as exception_output:
        simple_bandit_environment.step(action=3)

    # THEN
    assert "Invalid action=3. Action must be a member of [0, 1]" in str(
        exception_output.value
    )
