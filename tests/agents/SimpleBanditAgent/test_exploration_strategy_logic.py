from contextlib import contextmanager
from typing import List
import json

import numpy as np
import numpy.typing as npt
import pytest
from pathlib import Path

from gymnasium.spaces import Discrete
from gymnasium.utils.seeding import np_random

from algorithms.bandits.implementations.exploration.EpsilonGreedy import (
    EpsilonGreedy,
)
from algorithms.bandits.implementations.exploration.ExplorationExploitationContext import (
    ExplorationExploitationContext,
)
from algorithms.bandits.implementations.exploration.UpperConfidenceBound import (
    UpperConfidenceBound,
)
from utils.exceptions.logic_exceptions import ExplorationLogicException


@contextmanager
def not_raises():
    try:
        yield
    except Exception as e:
        raise AssertionError(f"Raised unexpected exception: {e}")


# GIVEN
@pytest.fixture
def full_run_ucb_values_path() -> Path:
    return Path(__file__).parent / "upper_confidence_bound_values.json"


# GIVEN
@pytest.fixture
def greedy_strategy() -> EpsilonGreedy:
    return EpsilonGreedy(epsilon=np.float64(0.0))


# GIVEN
@pytest.fixture
def ucb_strategy() -> UpperConfidenceBound:
    return UpperConfidenceBound(uncertainty_coefficient=np.float64(1))


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


# GIVEN
@pytest.fixture
def context_greedy(first_rng_16: np.random.Generator) -> ExplorationExploitationContext:
    context: ExplorationExploitationContext = ExplorationExploitationContext(
        rng=first_rng_16,
        action_space=Discrete(n=3, seed=16, start=0),
        q_values=np.array([2.0, 1.0, 0.0]),
        time_step=0,
    )
    return context


# GIVEN
@pytest.fixture
def contexts_ucb_action_counts() -> npt.NDArray[np.float64]:
    return np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 1],
            [1, 1, 2],
            [1, 1, 3],
            [1, 1, 4],
            [1, 1, 5],
            [1, 1, 6],
            [1, 1, 7],
        ],
        dtype=np.float64,
    )


# GIVEN
@pytest.fixture
def contexts_ucb(
    first_rng_16: np.random.Generator,
) -> List[ExplorationExploitationContext]:
    return [
        ExplorationExploitationContext(
            rng=first_rng_16,
            action_space=Discrete(n=3, seed=16, start=0),
            q_values=np.array([0.0, 0.0, 0.0]),
            time_step=0,
        ),
        ExplorationExploitationContext(
            rng=first_rng_16,
            action_space=Discrete(n=3, seed=16, start=0),
            q_values=np.array([2.0, 0.0, 0.0]),
            time_step=1,
        ),
        ExplorationExploitationContext(
            rng=first_rng_16,
            action_space=Discrete(n=3, seed=16, start=0),
            q_values=np.array([2.0, -1.0, 0.0]),
            time_step=2,
        ),
        ExplorationExploitationContext(
            rng=first_rng_16,
            action_space=Discrete(n=3, seed=16, start=0),
            q_values=np.array([2.0, -1.0, 3.0]),
            time_step=3,
        ),
        ExplorationExploitationContext(
            rng=first_rng_16,
            action_space=Discrete(n=3, seed=16, start=0),
            q_values=np.array([2.0, -1.0, 4.5]),
            time_step=4,
        ),
        ExplorationExploitationContext(
            rng=first_rng_16,
            action_space=Discrete(n=3, seed=16, start=0),
            q_values=np.array([2.0, -1.0, 4.3]),
            time_step=5,
        ),
        ExplorationExploitationContext(
            rng=first_rng_16,
            action_space=Discrete(n=3, seed=16, start=0),
            q_values=np.array([2.0, -1.0, 4.2]),
            time_step=6,
        ),
        ExplorationExploitationContext(
            rng=first_rng_16,
            action_space=Discrete(n=3, seed=16, start=0),
            q_values=np.array([2.0, -1.0, 3.6]),
            time_step=7,
        ),
        ExplorationExploitationContext(
            rng=first_rng_16,
            action_space=Discrete(n=3, seed=16, start=0),
            q_values=np.array([2.0, -1.0, 3]),
            time_step=8,
        ),
        ExplorationExploitationContext(
            rng=first_rng_16,
            action_space=Discrete(n=3, seed=16, start=0),
            q_values=np.array([2.0, -1.0, 2.85]),
            time_step=9,
        ),
    ]


def test_epsilon_greedy_negative_epsilon_throws_exception():
    # WHEN
    with pytest.raises(ExplorationLogicException) as exception_output:
        EpsilonGreedy(epsilon=np.float64(-1.0))

    # THEN
    assert "Invalid epsilon=-1.0. Epsilon must be between 0 and 1." in str(
        exception_output.value
    )


def test_pick_action_greedy(
    greedy_strategy: EpsilonGreedy, context_greedy: ExplorationExploitationContext
):
    # WHEN
    action_update_context = greedy_strategy.pick_action(context_greedy)
    action_greedy = action_update_context.action

    # THEN
    assert type(action_greedy) is int
    assert action_greedy == 0

    # WHEN
    context_greedy.q_values = np.array([-2.0, 1.0, 0.0])
    action_update_context = greedy_strategy.pick_action(context_greedy)
    action_greedy = action_update_context.action

    # THEN
    assert action_greedy == 1


def test_epsilon_greedy_string(greedy_strategy: EpsilonGreedy):
    # WHEN
    expected_string = "EpsilonGreedy(eps=0.0)"
    actual_string = greedy_strategy.__repr__()

    # THEN
    assert expected_string == actual_string


def test_ucb_string(ucb_strategy: UpperConfidenceBound):
    # WHEN
    expected_string = "UpperConfidenceBound(c=1.0)"
    actual_string = ucb_strategy.__repr__()

    # THEN
    assert expected_string == actual_string


def test_ucb_negative_uncertainty_coefficient_throws_exception():
    # WHEN
    with pytest.raises(ExplorationLogicException) as exception_output:
        UpperConfidenceBound(uncertainty_coefficient=np.float64(-1.0))

    # THEN
    assert (
        "Invalid uncertainty_coefficient=-1.0. uncertainty_coefficient must be bigger than 0."
        in str(exception_output.value)
    )


def test_ucb_pick_action_full_run(
    contexts_ucb: List[ExplorationExploitationContext],
    contexts_ucb_action_counts: npt.NDArray[np.float64],
    full_run_ucb_values_path: Path,
):
    # WHEN
    with full_run_ucb_values_path.open("r") as f:
        expected_rows: List = json.load(f)

    for index in range(len(contexts_ucb)):
        expected: dict = expected_rows[index]

        action_counts: npt.NDArray[np.float64] = contexts_ucb_action_counts[index]
        time_step: int = contexts_ucb[index].time_step
        q_values: npt.NDArray[np.float64] = contexts_ucb[index].q_values
        uncertainty_coefficient: np.float64 = np.float64(1.0)

        zero_indices = np.where(action_counts == 0)[0]

        if len(zero_indices) > 0:
            bonuses: npt.NDArray[np.float64] = np.array([], dtype=np.float64)
            scores: npt.NDArray[np.float64] = np.array([], dtype=np.float64)

            action_picked: int = zero_indices[0]
        else:
            bonuses: npt.NDArray[np.float64] = np.round(
                uncertainty_coefficient
                * np.sqrt(np.log(time_step + 1) / action_counts),
                2,
            )
            scores: npt.NDArray[np.float64] = q_values + bonuses

            action_picked: int = int(np.argmax(scores))

        # THEN
        assert expected["action_picked"] == action_picked
        assert np.allclose(expected["bonuses"], bonuses, atol=5e-2)
        assert np.allclose(expected["scores"], scores, atol=5e-2)
