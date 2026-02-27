import numpy as np
import pytest

from algorithms.bandits.implementations.action_value_update.ActionUpdateContext import (
    ActionUpdateContext,
)
from algorithms.bandits.implementations.action_value_update.AverageSampling import (
    AverageSampling,
)
from environments.custom_envs.BanditEnvs.KArmEnvironment import KArmEnvironment

# ==============
# Fixtures
# ==============


# GIVEN
@pytest.fixture
def stationary_env() -> KArmEnvironment:
    return KArmEnvironment(number_of_arms=2, seed=16, max_steps=10)


# GIVEN
@pytest.fixture
def average_sampling(stationary_env: KArmEnvironment) -> AverageSampling:
    action_value_strategy = AverageSampling()
    action_value_strategy._setup(env=stationary_env)
    return action_value_strategy


# GIVEN
@pytest.fixture
def action_update_context() -> ActionUpdateContext:
    return ActionUpdateContext(action=0)


# ==============
# Tests
# ==============


def test_average_sampling_update_action_values(
    average_sampling: AverageSampling, action_update_context: ActionUpdateContext
):
    # WHEN
    q_values = np.array([0.0, 0.0, 0.0])
    average_sampling.update_action_value(
        q_values=q_values,
        reward=np.float64(0.5),
        action_update_context=action_update_context,
    )

    # THEN
    assert average_sampling._action_counts[0] == 1
    assert q_values[0] == pytest.approx(0.5)

    # WHEN
    average_sampling.update_action_value(
        q_values=q_values,
        reward=np.float64(2.5),
        action_update_context=action_update_context,
    )

    # THEN
    assert average_sampling._action_counts[0] == 2
    assert q_values[0] == pytest.approx(1.5)


def test_average_sampling_repr(average_sampling: AverageSampling):
    # WHEN
    expected_string = "AverageSampling"
    actual_string = average_sampling.__repr__()

    # THEN
    assert expected_string == actual_string
