from contextlib import contextmanager

import numpy as np
import pytest

from algorithms.bandits.implementations.action_value.AverageSampling import (
    AverageSampling,
)
from algorithms.bandits.implementations.action_value.ERWAverageSampling import (
    ERWAverageSampling,
)
from environments.custom_envs.BanditEnvs.KArmEnvironment import KArmEnvironment
from utils.exceptions.logic_exceptions import ActionValueLogicException


@contextmanager
def not_raises():
    try:
        yield
    except Exception as e:
        raise AssertionError(f"Raised unexpected exception: {e}")


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
def erw_average_sampling(stationary_env: KArmEnvironment) -> ERWAverageSampling:
    action_value_strategy = ERWAverageSampling(alpha=1)
    action_value_strategy._setup(env=stationary_env)
    return action_value_strategy


def test_alpha_negative_throws_exception():
    # WHEN
    with pytest.raises(ActionValueLogicException) as exception_output:
        ERWAverageSampling(alpha=-2)

    # THEN
    assert "Invalid alpha=-2. Alpha must be between 1 and 0." in str(
        exception_output.value
    )


def test_average_sampling_update_action_values(average_sampling: AverageSampling):
    # WHEN
    q_values = np.array([0.0, 0.0, 0.0])
    average_sampling.update_action_value(q_values, action=0, reward=0.5)

    # THEN
    assert average_sampling._action_counts[0] == 1
    assert q_values[0] == pytest.approx(0.5)

    # WHEN
    average_sampling.update_action_value(q_values=q_values, action=0, reward=2.5)

    # THEN
    assert average_sampling._action_counts[0] == 2
    assert q_values[0] == pytest.approx(1.5)


def test_average_sampling_repr(average_sampling: AverageSampling):
    # WHEN
    expected_string = "AverageSampling()"
    actual_string = average_sampling.__repr__()

    # THEN
    assert expected_string == actual_string


def test_erw_average_sampling_update_action_values(
    erw_average_sampling: ERWAverageSampling,
):
    # WHEN
    q_values = np.array([0.0, 0.0, 0.0])
    erw_average_sampling.update_action_value(q_values, action=0, reward=0.5)

    # THEN
    assert q_values[0] == pytest.approx(0.5)

    # WHEN
    erw_average_sampling.update_action_value(q_values=q_values, action=0, reward=2.5)

    # THEN
    assert q_values[0] == pytest.approx(2.5)


def test_erw_average_sampling_repr(erw_average_sampling: ERWAverageSampling):
    # WHEN
    expected_string = "ERWAverageSampling(alpha=1)"
    actual_string = erw_average_sampling.__repr__()

    # THEN
    assert expected_string == actual_string
