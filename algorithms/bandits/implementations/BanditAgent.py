from functools import partial
from typing import Callable, List, Optional

import numpy as np
from algorithms.bandits.implementations.BaseBanditAgent import BaseBanditAgent
from algorithms.bandits.implementations.action_value.AverageSampling import (
    AverageSampling,
)
from algorithms.bandits.implementations.action_value.ActionValueStrategy import (
    ActionValueStrategy,
)
from algorithms.bandits.implementations.exploration.EpsilonGreedy import EpsilonGreedy
from algorithms.bandits.implementations.exploration.ExplorationExploitationStrategy import (
    ExplorationExploitationStrategy,
)
from environments.custom_envs.BanditEnvs.BaseBanditEnv import BaseBanditEnv
from gymnasium.utils.seeding import np_random

from utils.exceptions.logic_exceptions import AgentLogicException
from utils.logging.BaseLogger import BaseLogger


class BanditAgent(BaseBanditAgent):
    def __init__(
        self,
        env: BaseBanditEnv,
        seed: int,
        metrics: List[str],
        exploration_factory: Optional[
            Callable[..., ExplorationExploitationStrategy]
        ] = partial(EpsilonGreedy, epsilon=0.1),
        action_value_factory: Optional[Callable[..., ActionValueStrategy]] = partial(
            AverageSampling
        ),
    ):
        self._validate_input(seed=seed, metrics=metrics)

        self._env: BaseBanditEnv = env
        self._seed: int = seed
        self._metrics: List[str] = metrics
        self._n_arms: int = self._env.number_of_arms
        self._q_values: np.ndarray = np.zeros(shape=self._n_arms)

        self._exploration_strategy = exploration_factory()
        self._exploration_strategy.setup(env=self._env)

        self._action_value_strategy = action_value_factory()
        self._action_value_strategy.setup(env=self._env)

        self._reset_rng()

    # ==============
    # Properties
    # ==============

    @property
    def env(self) -> BaseBanditEnv:
        return self._env

    @property
    def metrics(self) -> List[str]:
        return self._metrics

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def n_arms(self) -> int:
        return self._n_arms

    @property
    def q_values(self) -> np.ndarray:
        return self._q_values

    # ==============
    # Public API
    # ==============

    def run_episode(self, logger: BaseLogger, log_every: int = 1):
        self._env.reset()

        total_reward: float = 0.0
        optimal_chosen_counter: int = 0
        total_steps: int = 0
        terminated = truncated = False

        while not terminated and not truncated:
            action: int = self._exploration_strategy.pick_action(
                rng=self.np_random,
                action_space=self.env.action_space,
                q_values=self._q_values,
            )

            _, reward, terminated, truncated, info = self._env.step(action=action)
            self._action_value_strategy.update_action_value(
                q_values=self._q_values,
                action=action,
                reward=reward,
            )

            total_reward += reward
            total_steps += 1
            optimal_chosen_counter += info["optimal_arm_chosen"]

            if total_steps % log_every == 0 or terminated or truncated:
                row = {
                    "step": total_steps,
                    "action": int(action),
                    "reward": float(reward),
                    "total_reward": float(total_reward),
                    "average_reward": float(total_reward) / total_steps,
                    "optimal_chosen_counter": float(optimal_chosen_counter),
                    "optimal_chosen_percentage": (
                        float(optimal_chosen_counter) / total_steps
                        if total_steps
                        else 0.0
                    ),
                }
                logger.log(row=row)

        return {
            "episode_reward": total_reward,
            "steps": total_steps,
            "optimal_chosen_percentage": (
                optimal_chosen_counter / total_steps if total_steps else 0.0
            ),
        }

    # ==============
    # Internals
    # ==============

    def _reset_rng(self):
        self.np_random, _ = np_random(self._seed)

    def _set_seed(self, new_seed: int):
        self._seed = new_seed
        self._reset_rng()

    def _validate_input(self, seed: int, metrics: List[str]):
        if seed < 0:
            raise AgentLogicException(
                f"Invalid seed={seed}. This number must be positive."
            )
        if len(metrics) == 0:
            raise AgentLogicException(
                f"Invalid metrics={metrics}. The array should not be empty."
            )

    def __str__(self):
        return f"{self.__class__.__name__}(seed={self._seed})"

    def __repr__(self):
        return f"{self.__class__.__name__}(seed={self._seed},\n\tenv={self.env.__repr__()},\n\taction_value={self._action_value_strategy.__repr__()},\n\texploration={self._exploration_strategy.__repr__()}\n)"
