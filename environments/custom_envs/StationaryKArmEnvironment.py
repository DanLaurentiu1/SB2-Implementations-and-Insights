import numpy as np
from gymnasium.utils.seeding import np_random
from gymnasium import Env
from gymnasium.spaces import Discrete

from utils.exceptions.logic_exceptions import EnvironmentLogicException


class StationaryKArmEnvironment(Env):
    def __init__(self, number_of_arms: int = 10, seed: int = 16, max_steps: int = 1000):
        self._validate_input(
            number_of_arms=number_of_arms, seed=seed, max_steps=max_steps
        )

        self.terminated = False
        self.truncated = False
        self.number_of_arms = number_of_arms
        self.seed = seed
        self.pulls = 0
        self.max_steps = max_steps
        self.np_random, _ = np_random(self.seed)
        self.observation_space = Discrete(1, seed=self.seed)
        self.action_space = Discrete(n=self.number_of_arms, seed=self.seed, start=0)
        self._get_new_arms()

    def _get_new_arms(self):
        self.arm_means = self.np_random.normal(
            loc=0.0, scale=1.0, size=self.number_of_arms
        )
        self.optimal_arm = int(np.argmax(self.arm_means))

    def _get_obs(self):
        return np.float64(1)

    def _get_info(self, optimal_arm_chosen: bool):
        return {"optimal_arm_chosen": optimal_arm_chosen}

    def _ensure_not_terminated(self):
        if self.terminated:
            raise EnvironmentLogicException(
                f"step() called after rollout termination. call reset() first."
            )

    def _validate_input(self, number_of_arms: int, seed: int, max_steps: int):
        if number_of_arms < 1:
            raise EnvironmentLogicException(
                f"Invalid number of arms={number_of_arms}. This number must be positive and bigger than 0."
            )
        if max_steps < 1:
            raise EnvironmentLogicException(
                f"Invalid number of max_steps={max_steps}. This number must be positive and bigger than 0."
            )
        if seed < 0:
            raise EnvironmentLogicException(
                f"Invalid seed={seed}. This number must be positive."
            )

    def _validate_action(self, action: int):
        if not (0 <= action < self.number_of_arms):
            raise EnvironmentLogicException(
                f"Invalid action={action}. Action must be a member of [0, {self.number_of_arms - 1}]"
            )

    def reset(self, *, seed=None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed = seed
            self.np_random, _ = np_random(self.seed)
            self._get_new_arms()

        self.terminated = False
        self.pulls = 0
        info = {"optimal_arm": self.optimal_arm}
        return self._get_obs(), info

    def step(self, action: int):
        self._ensure_not_terminated()
        self._validate_action(action=action)

        self.pulls += 1
        if self.pulls == self.max_steps:
            self.terminated = True
        is_optimal = action == self.optimal_arm
        reward = float(self.np_random.normal(loc=self.arm_means[action], scale=1.0))
        return (
            self._get_obs(),
            reward,
            self.terminated,
            self.truncated,
            self._get_info(optimal_arm_chosen=is_optimal),
        )
