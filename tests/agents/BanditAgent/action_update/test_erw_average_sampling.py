import numpy as np
import pytest

from algorithms.bandits.implementations.action_value_update.ActionUpdateContext import (
    ActionUpdateContext,
)
from algorithms.bandits.implementations.action_value_update.ERWAverageSampling import (
    ERWAverageSampling,
)
from environments.custom_envs.BanditEnvs.KArmEnvironment import KArmEnvironment
from utils.exceptions.logic_exceptions import ActionUpdateLogicException


# ==============
# Fixtures
# ==============


# GIVEN
@pytest.fixture
def stationary_env() -> KArmEnvironment:
    return KArmEnvironment(number_of_arms=2, seed=16, max_steps=10)


# GIVEN
@pytest.fixture
def erw_average_sampling(stationary_env: KArmEnvironment) -> ERWAverageSampling:
    action_value_strategy = ERWAverageSampling(alpha=np.float64(1))
    action_value_strategy._setup(env=stationary_env)
    return action_value_strategy


# GIVEN
@pytest.fixture
def action_update_context() -> ActionUpdateContext:
    return ActionUpdateContext(action=0)


# ==============
# Tests
# ==============


def test_alpha_negative_throws_exception():
    # WHEN
    with pytest.raises(ActionUpdateLogicException) as exception_output:
        ERWAverageSampling(alpha=np.float64(-2))

    # THEN
    assert "Invalid alpha=-2.0. Alpha must be between 1 and 0." in str(
        exception_output.value
    )


def test_erw_average_sampling_update_action_values(
    erw_average_sampling: ERWAverageSampling, action_update_context: ActionUpdateContext
):
    # WHEN
    q_values = np.array([0.0, 0.0, 0.0])
    erw_average_sampling.update_action_value(
        q_values=q_values,
        reward=np.float64(0.5),
        action_update_context=action_update_context,
    )

    # THEN
    assert q_values[0] == pytest.approx(0.5)

    # WHEN
    erw_average_sampling.update_action_value(
        q_values=q_values,
        reward=np.float64(2.5),
        action_update_context=action_update_context,
    )

    # THEN
    assert q_values[0] == pytest.approx(2.5)


def test_erw_average_sampling_repr(erw_average_sampling: ERWAverageSampling):
    # WHEN
    expected_string = "ERWAverageSampling(alpha=1.0)"
    actual_string = erw_average_sampling.__repr__()

    # THEN
    assert expected_string == actual_string
