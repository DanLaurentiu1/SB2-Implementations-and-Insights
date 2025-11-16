import json
from pathlib import Path
import numpy as np
import pytest
from algorithms.bandits.implementations.BanditAgent import BanditAgent
from environments.custom_envs.StationaryKArmEnvironment import StationaryKArmEnvironment
from utils.logging.FakeLogger import FakeLogger


# GIVEN
@pytest.fixture
def simple_env() -> StationaryKArmEnvironment:
    return StationaryKArmEnvironment(number_of_arms=3, seed=16, max_steps=5)


@pytest.fixture
def agent(simple_env: StationaryKArmEnvironment) -> BanditAgent:
    return BanditAgent(
        env=simple_env,
        epsillon=0.1,
        seed=16,
        metrics=[
            "step",
            "action",
            "reward",
            "total_reward",
            "average_reward",
            "optimal_chosen_percentage",
        ],
    )


def test_constructor_initializes_fields(
    agent: BanditAgent, simple_env: StationaryKArmEnvironment
):
    # THEN
    assert agent.env is simple_env
    assert agent.epsillon == pytest.approx(0.1)
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
    assert agent.q_values.shape[0] == simple_env.number_of_arms
    assert np.all(agent.q_values == 0)
    assert isinstance(agent.action_freq, np.ndarray)
    assert agent.action_freq.shape[0] == simple_env.number_of_arms
    assert np.all(agent.action_freq == 0)


def test_set_seed_changes_rng_state(agent: BanditAgent):
    # WHEN
    random_1 = agent.np_random.random()
    agent.set_seed(42)
    random_2 = agent.np_random.random()
    agent.set_seed(42)
    random_3 = agent.np_random.random()

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
    assert agent.get_metrics() == metrics


def test_update_action_values(agent: BanditAgent):
    # WHEN
    agent._update_action_value(action=0, reward=0.5)

    # THEN
    assert agent.action_freq[0] == 1
    assert agent.q_values[0] == pytest.approx(0.5)

    # WHEN
    agent._update_action_value(action=0, reward=2.5)

    # THEN
    assert agent.action_freq[0] == 2
    assert agent.q_values[0] == pytest.approx(1.5)


def test_run_episode_logs_and_returns(agent: BanditAgent):
    logger = FakeLogger()

    # WHEN
    out = agent.run_episode(logger=logger, log_every=1)

    # THEN
    assert "episode_reward" in out
    assert "steps" in out
    assert "optimal_chosen_percentage" in out
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


def test_pick_action(agent: BanditAgent):
    # WHEN
    agent.q_values = np.array([2.0, 1.0, 0.0])
    agent.epsillon = 0.0
    action_greedy = agent._pick_action()

    # THEN
    assert action_greedy == 0

    # WHEN
    agent.q_values = np.array([-2.0, 1.0, 0.0])
    action_greedy = agent._pick_action()

    # THEN
    assert action_greedy == 1


def test_full_run(agent: BanditAgent):
    # WHEN
    logger = FakeLogger()

    total_reward = 0.0
    optimal_chosen_counter = total_steps = 0
    terminated = truncated = False

    while not terminated and not truncated:
        action = agent._pick_action()

        _, reward, terminated, truncated, info = agent.env.step(action=action)
        agent._update_action_value(action=action, reward=reward)

        total_reward += reward
        total_steps += 1
        optimal_chosen_counter += info["optimal_arm_chosen"]

        row = {
            "step": total_steps,
            "action": int(action),
            "reward": float(reward),
            "total_reward": float(total_reward),
            "average_reward": float(total_reward) / total_steps,
            "optimal_chosen_counter": float(optimal_chosen_counter),
            "optimal_chosen_percentage": (
                float(optimal_chosen_counter) / total_steps if total_steps else 0.0
            ),
            "q_values": agent.q_values.tolist(),
            "action_freq": agent.action_freq.tolist(),
        }

        logger.log(row=row)

    json_path = Path("tests/agents/SimpleBanditAgent/agent_16_behaviour_values.json")
    with json_path.open("r") as f:
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
        assert actual["q_values"] == pytest.approx(expected["q_values"], abs=5e-2)
        assert actual["action_freq"] == expected["action_freq"]
