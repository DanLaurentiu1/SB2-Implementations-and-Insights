import json
from pathlib import Path
from typing import List
import pytest
import numpy as np

from algorithms.bandits.implementations.action_value_update.ActionUpdateContext import (
    ActionUpdateContext,
)
from algorithms.bandits.implementations.action_value_update.PreferenceGradientSampling import (
    PreferenceGradientSampling,
)
from environments.custom_envs.BanditEnvs.KArmEnvironment import KArmEnvironment
from utils.exceptions.logic_exceptions import ActionUpdateLogicException

# ==============
# Fixtures
# ==============


# GIVEN
@pytest.fixture
def full_run_values_path() -> Path:
    return Path(__file__).parent / "data" / "preference_gradient_full_run_values.json"


# GIVEN
@pytest.fixture
def full_run_values_path_no_baseline() -> Path:
    return (
        Path(__file__).parent / "data" / "preference_gradient_full_run_no_baseline.json"
    )


# GIVEN
@pytest.fixture
def env() -> KArmEnvironment:
    return KArmEnvironment(number_of_arms=3)


# GIVEN
@pytest.fixture
def preference_sampling_with_baseline(
    env: KArmEnvironment,
) -> PreferenceGradientSampling:
    action_value_strategy = PreferenceGradientSampling(
        alpha=np.float64(1.0), baseline=True
    )
    action_value_strategy._setup(env=env)
    return action_value_strategy


# GIVEN
@pytest.fixture
def preference_sampling_without_baseline(
    env: KArmEnvironment,
) -> PreferenceGradientSampling:
    action_value_strategy = PreferenceGradientSampling(
        alpha=np.float64(1.0), baseline=False
    )
    action_value_strategy._setup(env=env)
    return action_value_strategy


# GIVEN
@pytest.fixture
def action_update_context() -> ActionUpdateContext:
    return ActionUpdateContext(
        action=0, probabilities=np.array([0.5, 0.3, 0.2], dtype=np.float64)
    )


# GIVEN
@pytest.fixture
def action_update_context_no_probabilities() -> ActionUpdateContext:
    return ActionUpdateContext(action=0)


# ==============
# Tests
# ==============


def test_preference_gradient_negative_alpha_throws_exception(
    preference_sampling_with_baseline: PreferenceGradientSampling,
    action_update_context_no_probabilities: ActionUpdateContext,
):
    with pytest.raises(ActionUpdateLogicException) as exception_output:
        preference_sampling_with_baseline.update_action_value(
            q_values=np.array([2, 1.5, 1], dtype=np.float64),
            reward=np.float64(1.5),
            action_update_context=action_update_context_no_probabilities,
        )

    assert (
        "PreferenceGradientSampling requires probabilities in the ActionUpdateContext."
        in str(exception_output.value)
    )


def test_preference_gradient_string_representation(
    preference_sampling_with_baseline: PreferenceGradientSampling,
):
    expected_string = "PGS(a=1.0,b=True)"
    actual_string = preference_sampling_with_baseline.__repr__()

    assert expected_string == actual_string


def test_preference_gradient_string_representation_no_baseline(
    preference_sampling_without_baseline: PreferenceGradientSampling,
):
    expected_string = "PGS(a=1.0,b=False)"
    actual_string = preference_sampling_without_baseline.__repr__()

    assert expected_string == actual_string


def test_preference_gradient_update_baseline(
    preference_sampling_with_baseline: PreferenceGradientSampling,
):
    rewards = np.random.uniform(low=0, high=100, size=10000)
    for i in range(len(rewards)):
        preference_sampling_with_baseline._update_baseline(
            latest_reward=np.float64(rewards[i])
        )
        assert preference_sampling_with_baseline._baseline == pytest.approx(
            np.mean(rewards[: i + 1])
        )


def test_preference_gradient_validate_input():
    with pytest.raises(ActionUpdateLogicException) as exception_output:
        PreferenceGradientSampling(alpha=np.float64(-1), baseline=True)

    assert "Invalid alpha=-1.0. Alpha must be bigger than 0." in str(
        exception_output.value
    )


def test_preference_gradient_q_values_learning_formula(
    preference_sampling_with_baseline: PreferenceGradientSampling,
    action_update_context: ActionUpdateContext,
):
    actual_q_values = np.array([2, 1.5, 1])
    expected_q_values = np.array([2.75, 1.05, 0.7])

    preference_sampling_with_baseline.update_action_value(
        q_values=actual_q_values,
        reward=np.float64(1.5),
        action_update_context=action_update_context,
    )

    assert np.allclose(actual_q_values, expected_q_values)


def test_preference_gradient_full_run(
    preference_sampling_with_baseline: PreferenceGradientSampling,
    full_run_values_path: Path,
):
    # WHEN
    with full_run_values_path.open("r") as f:
        expected_rows: List = json.load(f)

    for current_rows in expected_rows:
        input_context = ActionUpdateContext(
            action=current_rows["input"]["action"],
            probabilities=np.array(
                current_rows["input"]["probabilities"], dtype=np.float64
            ),
        )
        actual_q_values = np.array(current_rows["input"]["q-values"], dtype=np.float64)
        current_reward = current_rows["input"]["reward"]
        expected_q_values = np.array(
            current_rows["output"]["q-values"], dtype=np.float64
        )
        expected_baseline = current_rows["output"]["baseline"]

        preference_sampling_with_baseline.update_action_value(
            q_values=actual_q_values,
            reward=current_reward,
            action_update_context=input_context,
        )

        actual_baseline = preference_sampling_with_baseline._baseline

        assert actual_baseline == pytest.approx(expected_baseline, abs=2e-3)
        assert np.allclose(actual_q_values, expected_q_values, atol=2e-3)


def test_preference_gradient_no_baseline_full_run(
    preference_sampling_without_baseline: PreferenceGradientSampling,
    full_run_values_path_no_baseline: Path,
):
    # WHEN
    with full_run_values_path_no_baseline.open("r") as f:
        expected_rows: List = json.load(f)

    for current_rows in expected_rows:
        input_context = ActionUpdateContext(
            action=current_rows["input"]["action"],
            probabilities=np.array(
                current_rows["input"]["probabilities"], dtype=np.float64
            ),
        )
        actual_q_values = np.array(current_rows["input"]["q-values"], dtype=np.float64)
        current_reward = current_rows["input"]["reward"]
        expected_q_values = np.array(
            current_rows["output"]["q-values"], dtype=np.float64
        )
        preference_sampling_without_baseline.update_action_value(
            q_values=actual_q_values,
            reward=current_reward,
            action_update_context=input_context,
        )

        assert np.allclose(actual_q_values, expected_q_values, atol=2e-3)
