import json
from functools import partial
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from algorithms.bandits.implementations.action_value_initialization.NormalActionValueInitialization import (
    NormalActionValueInitialization,
)
from algorithms.bandits.implementations.action_value_initialization.OptimisticActionValueInitialization import (
    OptimisticActionValueInitialization,
)
from algorithms.bandits.implementations.action_value_update.AverageSampling import (
    AverageSampling,
)
from algorithms.bandits.implementations.BanditAgent import BanditAgent
from algorithms.bandits.implementations.action_value_update.ERWAverageSampling import (
    ERWAverageSampling,
)
from algorithms.bandits.implementations.exploration.EpsilonGreedy import (
    EpsilonGreedy,
)
from algorithms.bandits.implementations.exploration.ExplorationExploitationContext import (
    ExplorationExploitationContext,
)
from environments.custom_envs.BanditEnvs.KArmEnvironment import KArmEnvironment
from utils.exceptions.logic_exceptions import AgentLogicException
from utils.logging.FakeLogger import FakeLogger

# ==============
# Fixtures
# ==============


# GIVEN
@pytest.fixture
def stationary_env() -> KArmEnvironment:
    return KArmEnvironment(
        number_of_arms=3,
        seed=16,
        max_steps=5,
    )


# GIVEN
@pytest.fixture
def agent(stationary_env: KArmEnvironment) -> BanditAgent:
    return BanditAgent(
        env=stationary_env,
        seed=16,
        metrics=[
            "step",
            "action",
            "reward",
            "total_reward",
            "average_reward",
            "optimal_chosen_percentage",
        ],
        exploration_factory=partial(EpsilonGreedy, epsilon=np.float64(0.1)),
        action_value_update_factory=partial(AverageSampling),
        action_value_initialization_factory=partial(NormalActionValueInitialization),
    )


# GIVEN
@pytest.fixture
def context(agent: BanditAgent) -> ExplorationExploitationContext:
    context: ExplorationExploitationContext = ExplorationExploitationContext(
        rng=agent._np_random,
        action_space=agent.env.action_space,
        q_values=agent.q_values,
        time_step=0,
    )
    return context


# GIVEN
@pytest.fixture
def optimistic_agent(stationary_env: KArmEnvironment) -> BanditAgent:
    return BanditAgent(
        env=stationary_env,
        seed=16,
        metrics=[
            "step",
            "action",
            "reward",
            "total_reward",
            "average_reward",
            "optimal_chosen_percentage",
        ],
        exploration_factory=partial(EpsilonGreedy, epsilon=np.float64(0.1)),
        action_value_update_factory=partial(ERWAverageSampling, epsilon=0.1),
        action_value_initialization_factory=partial(
            OptimisticActionValueInitialization
        ),
    )


# GIVEN
@pytest.fixture
def full_run_values_json_path() -> Path:
    return Path(__file__).parent / "data" / "agent_16_behaviour_values.json"


# GIVEN
@pytest.fixture
def full_run_values_json_path_logging_skip() -> Path:
    return (
        Path(__file__).parent / "data" / "agent_16_behaviour_values_logging_skip.json"
    )


# ==============
# Tests
# ==============


def test_agent_invalid_seed_throws_exception(stationary_env: KArmEnvironment):
    with pytest.raises(AgentLogicException) as exception_output:
        BanditAgent(
            env=stationary_env,
            seed=-12,
            metrics=[
                "step",
                "action",
                "reward",
                "total_reward",
                "average_reward",
                "optimal_chosen_percentage",
            ],
            exploration_factory=partial(EpsilonGreedy, epsilon=np.float64(0.1)),
            action_value_update_factory=partial(AverageSampling),
        )

    assert "Invalid seed=-12. This number must be positive." in str(
        exception_output.value
    )


def test_agent_invalid_metrics_throws_exception(stationary_env: KArmEnvironment):
    with pytest.raises(AgentLogicException) as exception_output:
        BanditAgent(
            env=stationary_env,
            seed=12,
            metrics=[],
            exploration_factory=partial(EpsilonGreedy, epsilon=np.float64(0.1)),
            action_value_update_factory=partial(AverageSampling),
        )

    assert "Invalid metrics=[]. The array should not be empty." in str(
        exception_output.value
    )


def test_constructor_initializes_fields(
    agent: BanditAgent, stationary_env: KArmEnvironment
):
    # THEN
    assert agent.env is stationary_env
    assert agent.seed == 16
    assert agent.metrics == [
        "step",
        "action",
        "reward",
        "total_reward",
        "average_reward",
        "optimal_chosen_percentage",
    ]
    assert agent.n_arms == 3
    assert isinstance(agent.q_values, np.ndarray)
    assert agent.q_values.shape[0] == stationary_env.number_of_arms
    assert np.all(agent.q_values == 0)

    assert isinstance(agent._exploration_strategy, EpsilonGreedy)
    assert agent._exploration_strategy.epsilon == 0.1

    assert isinstance(agent._action_value_update_strategy, AverageSampling)
    assert isinstance(agent._action_value_update_strategy._action_counts, np.ndarray)
    assert (
        agent._action_value_update_strategy._action_counts.shape[0]
        == stationary_env.number_of_arms
    )
    assert np.all(agent._action_value_update_strategy._action_counts == 0)


def test_set_seed_changes_rng_state(agent: BanditAgent):
    # WHEN
    random_1 = agent._np_random.random()
    agent._set_seed(42)
    random_2 = agent._np_random.random()
    agent._set_seed(42)
    random_3 = agent._np_random.random()

    # THEN
    assert random_1 != random_2
    assert random_2 == random_3


def test_get_metrics(agent: BanditAgent):
    # WHEN
    metrics = [
        "step",
        "action",
        "reward",
        "total_reward",
        "average_reward",
        "optimal_chosen_percentage",
    ]

    # THEN
    assert agent.metrics == metrics


def test_run_episode_logs_and_returns(agent: BanditAgent):
    logger = FakeLogger()

    # WHEN
    agent.run_episode(logger=logger, log_every=1)

    # THEN
    assert len(logger.rows) == agent.env.max_steps

    # WHEN
    first_row = logger.rows[0]

    # THEN
    assert "step" in first_row
    assert "action" in first_row
    assert "reward" in first_row
    assert "total_reward" in first_row
    assert "average_reward" in first_row
    assert "optimal_chosen_percentage" in first_row


def test_agent_repr(agent: BanditAgent):
    # WHEN
    expected_string = "BanditAgent(s=16,\n\tenv=KArm(\n\tdft=NoDrift,\n\tr=GaussianReward(var=1.0),\n\ts=16,\n\tarms=3\n),\n\ta_v=AverageSampling,\n\tinit=NormalAVInit,\n\texpl=EpsilonGreedy(eps=0.1)\n)"
    actual_string = agent.__repr__()

    # THEN
    assert expected_string == actual_string


def test_agent_str(agent: BanditAgent):
    # WHEN
    expected_string = (
        "BanditAgent(s=16,expl=EpsilonGreedy,a_v=AverageSampling,init=NormalAVInit)"
    )
    actual_string = agent.__str__()

    # THEN
    assert expected_string == actual_string


def test_full_run_simple(agent: BanditAgent, full_run_values_json_path: Path):
    # WHEN
    logger = FakeLogger()
    agent.run_episode(logger=logger, log_every=1)

    with full_run_values_json_path.open("r") as f:
        expected_rows = json.load(f)

    # THEN
    assert len(logger.rows) == len(expected_rows)
    for actual, expected in zip(logger.get_rows(), expected_rows):
        assert actual["step"] == expected["step"]
        assert actual["action"] == expected["action"]
        assert actual["reward"] == pytest.approx(expected["reward"], abs=5e-2)
        assert actual["total_reward"] == pytest.approx(
            expected["total_reward"], abs=5e-2
        )
        assert actual["average_reward"] == pytest.approx(
            expected["average_reward"], abs=5e-2
        )
        assert actual["optimal_chosen_counter"] == pytest.approx(
            expected["optimal_chosen_counter"], abs=5e-2
        )
        assert actual["optimal_chosen_percentage"] == pytest.approx(
            expected["optimal_chosen_percentage"], abs=5e-2
        )


def test_full_run_simple_skip_logging(
    agent: BanditAgent, full_run_values_json_path_logging_skip: Path
):
    # WHEN
    logger = FakeLogger()
    agent.run_episode(logger=logger, log_every=2)

    with full_run_values_json_path_logging_skip.open("r") as f:
        expected_rows = json.load(f)

    # THEN
    assert len(logger.rows) == len(expected_rows)
    for actual, expected in zip(logger.get_rows(), expected_rows):
        assert actual["step"] == expected["step"]
        assert actual["action"] == expected["action"]
        assert actual["reward"] == pytest.approx(expected["reward"], abs=5e-2)
        assert actual["total_reward"] == pytest.approx(
            expected["total_reward"], abs=5e-2
        )
        assert actual["average_reward"] == pytest.approx(
            expected["average_reward"], abs=5e-2
        )
        assert actual["optimal_chosen_counter"] == pytest.approx(
            expected["optimal_chosen_counter"], abs=5e-2
        )
        assert actual["optimal_chosen_percentage"] == pytest.approx(
            expected["optimal_chosen_percentage"], abs=5e-2
        )


def test_full_run_advanced(
    agent: BanditAgent,
    full_run_values_json_path: Path,
    context: ExplorationExploitationContext,
):
    action_value_strategy = cast(AverageSampling, agent._action_value_update_strategy)

    # WHEN
    logger = FakeLogger()

    agent.run_episode(logger=logger, log_every=1)

    with full_run_values_json_path.open("r") as f:
        expected_rows = json.load(f)

    # THEN
    assert len(logger.rows) == len(expected_rows)
    for actual, expected in zip(logger.get_rows(), expected_rows):
        assert actual["step"] == expected["step"]
        assert actual["action"] == expected["action"]
        assert actual["reward"] == pytest.approx(expected["reward"], abs=5e-2)
        assert actual["total_reward"] == pytest.approx(
            expected["total_reward"], abs=5e-2
        )
        assert actual["average_reward"] == pytest.approx(
            expected["average_reward"], abs=5e-2
        )
        assert actual["optimal_chosen_counter"] == pytest.approx(
            expected["optimal_chosen_counter"], abs=5e-2
        )
        assert actual["optimal_chosen_percentage"] == pytest.approx(
            expected["optimal_chosen_percentage"], abs=5e-2
        )

    assert agent.q_values == pytest.approx(expected_rows[-1]["q_values"], abs=5e-2)
    assert np.allclose(
        action_value_strategy.action_counts, np.array(expected_rows[-1]["action_freq"])
    )
