from contextlib import contextmanager

import numpy as np
import pytest
from gymnasium.spaces import Discrete
from gymnasium.utils.seeding import np_random

from algorithms.bandits.implementations.exploration.EpsilonGreedy import EpsilonGreedy
from utils.exceptions.logic_exceptions import ExplorationLogicException


@contextmanager
def not_raises():
    try:
        yield
    except Exception as e:
        raise AssertionError(f"Raised unexpected exception: {e}")


@pytest.fixture
def greedy_strategy() -> EpsilonGreedy:
    return EpsilonGreedy(epsilon=0.0)


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


def test_epsilon_greedy_negative_epsilon_throws_exception():
    # WHEN
    with pytest.raises(ExplorationLogicException) as exception_output:
        EpsilonGreedy(epsilon=-1.0)

    # THEN
    assert "Invalid epsilon=-1.0. Epsilon must be between 0 and 1." in str(
        exception_output.value
    )


def test_pick_action(greedy_strategy: EpsilonGreedy, first_rng_16: np.random.Generator):
    # WHEN
    action_greedy = greedy_strategy.pick_action(
        rng=first_rng_16,
        action_space=Discrete(n=3, seed=16, start=0),
        q_values=np.array([2.0, 1.0, 0.0]),
    )

    # THEN
    assert type(action_greedy) is int
    assert action_greedy == 0

    # WHEN
    action_greedy = greedy_strategy.pick_action(
        rng=first_rng_16,
        action_space=Discrete(n=3, seed=16, start=0),
        q_values=np.array([-2.0, 1.0, 0.0]),
    )

    # THEN
    assert action_greedy == 1


def test_epsilon_greedy_string(greedy_strategy: EpsilonGreedy):
    # WHEN
    expected_string = "EpsilonGreedy(eps=0.0)"
    actual_string = greedy_strategy.__repr__()

    # THEN
    assert expected_string == actual_string
