from typing import List

import numpy as np
from environments.custom_envs.StationaryKArmEnvironment import StationaryKArmEnvironment
from gymnasium.utils.seeding import np_random

from utils.logging.BaseLogger import BaseLogger


class BanditAgent:
    def __init__(
        self,
        env: StationaryKArmEnvironment,
        epsillon: float,
        seed: int,
        metrics: List[str],
    ):
        self.env = env
        self.epsillon = epsillon
        self.seed = seed
        self.metrics = metrics
        self.n_arms = self.env.number_of_arms
        self.q_values = np.zeros(shape=self.n_arms)
        self.action_freq = np.zeros(shape=self.n_arms, dtype=int)

        self._reset_rng()

    def _reset_rng(self):
        self.np_random, _ = np_random(self.seed)

    def _update_action_value(self, action: int, reward: float):
        self.action_freq[action] += 1
        old_q_values = self.q_values[action]
        self.q_values[action] = old_q_values + (
            (1 / self.action_freq[action]) * (reward - old_q_values)
        )

    def _pick_action(self):
        if self.np_random.random() < self.epsillon:
            action = int(self.env.action_space.sample())
        else:
            action = int(np.argmax(self.q_values))
        return action

    def get_metrics(self):
        return self.metrics

    def set_seed(self, new_seed: int):
        self.seed = new_seed
        self._reset_rng()

    def run_episode(self, logger: BaseLogger, log_every: int = 1):
        self.env.reset()

        total_reward = 0.0
        optimal_chosen_counter = total_steps = 0
        terminated = truncated = False

        while not terminated and not truncated:
            action = self._pick_action()

            _, reward, terminated, truncated, info = self.env.step(action=action)
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
