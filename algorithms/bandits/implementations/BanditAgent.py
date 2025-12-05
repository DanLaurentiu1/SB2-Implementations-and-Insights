from typing import List

import numpy as np
from algorithms.bandits.implementations.BaseBanditAgent import BaseBanditAgent
from algorithms.bandits.implementations.exploration.EpsilonGreedy import EpsilonGreedy
from algorithms.bandits.implementations.exploration.ExplorationExploitationStrategy import (
    ExplorationExploitationStrategy,
)
from environments.custom_envs.BanditEnvs.BaseBanditEnv import BaseBanditEnv
from gymnasium.utils.seeding import np_random

from utils.logging.BaseLogger import BaseLogger


class BanditAgent(BaseBanditAgent):
    def __init__(
        self,
        env: BaseBanditEnv,
        seed: int,
        metrics: List[str],
        exploration_strategy: ExplorationExploitationStrategy | None = None,
    ):
        self._env: BaseBanditEnv = env
        self._seed: int = seed
        self._metrics: List[str] = metrics
        self._n_arms: int = self._env.number_of_arms
        self._q_values: np.ndarray = np.zeros(shape=self._n_arms)
        self._action_freq: np.ndarray = np.zeros(shape=self._n_arms, dtype=int)

        if not exploration_strategy:
            self._exploration_strategy = EpsilonGreedy(epsilon=0.1)
        else:
            self._exploration_strategy = exploration_strategy

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
    def epsilon(self) -> float:
        return self._exploration_strategy.epsilon

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def n_arms(self) -> int:
        return self._n_arms

    @property
    def q_values(self) -> np.ndarray:
        return self._q_values

    @property
    def action_freq(self) -> np.ndarray:
        return self._action_freq

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
            action: int = self._exploration_strategy._pick_action(
                rng=self.np_random,
                action_space=self.env.action_space,
                q_values=self._q_values,
            )

            _, reward, terminated, truncated, info = self._env.step(action=action)
            self._update_action_value(action=action, reward=reward)

            total_reward += reward
            total_steps += 1
            optimal_chosen_counter += info["optimal_arm_chosen"]

            if total_steps % log_every == 0 or terminated or truncated:
                row = {
                    "step": total_steps,
                    "action": action,
                    "reward": reward,
                    "total_reward": total_reward,
                    "average_reward": total_reward / total_steps,
                    "optimal_chosen_percentage": optimal_chosen_counter / total_steps,
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

    def _update_action_value(self, action: int, reward: float):
        self._action_freq[action] += 1
        old_q_values = self._q_values[action]
        self._q_values[action] = old_q_values + (
            (1 / self._action_freq[action]) * (reward - old_q_values)
        )

    def _set_seed(self, new_seed: int):
        self._seed = new_seed
        self._reset_rng()

    def __str__(self):
        return f"Agent(seed={self._seed}, eps={self._epsilon})"
