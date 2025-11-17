import numpy as np
from gymnasium.utils.seeding import np_random
from gymnasium import Env, Space
from gymnasium.spaces import Discrete

from environments.custom_envs.BanditEnvs.BaseBanditEnv import BaseBanditEnv
from utils.exceptions.logic_exceptions import EnvironmentLogicException


class StationaryKArmEnvironment(Env, BaseBanditEnv):
    def __init__(self, number_of_arms: int = 10, seed: int = 16, max_steps: int = 1000):
        self._validate_input(
            number_of_arms=number_of_arms, seed=seed, max_steps=max_steps
        )

        self._terminated = False
        self._truncated = False
        self._number_of_arms = number_of_arms
        self._seed = seed
        self._pulls = 0
        self._max_steps = max_steps
        self._np_random, _ = np_random(self._seed)
        self._observation_space = Discrete(1, seed=self._seed)
        self._action_space = Discrete(n=self._number_of_arms, seed=self._seed, start=0)
        self._get_new_arms()

    def _get_new_arms(self):
        self._arm_means = self._np_random.normal(
            loc=0.0, scale=1.0, size=self._number_of_arms
        )
        self._optimal_arm = int(np.argmax(self._arm_means))

    def _get_obs(self):
        return np.float64(1)

    def _get_info(self, optimal_arm_chosen: bool):
        return {"optimal_arm_chosen": optimal_arm_chosen}

    def _ensure_not_terminated(self):
        if self._terminated:
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
        if not (0 <= action < self._number_of_arms):
            raise EnvironmentLogicException(
                f"Invalid action={action}. Action must be a member of [0, {self._number_of_arms - 1}]"
            )

    @property
    def number_of_arms(self) -> int:
        return self._number_of_arms

    @property
    def max_steps(self) -> int:
        return self._max_steps

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def action_space(self) -> Space:
        return self._action_space

    @property
    def observation_space(self) -> Space:
        return self._observation_space

    def reset(self, *, seed=None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
            self._np_random, _ = np_random(self._seed)
            self._get_new_arms()

        self._terminated = False
        self._pulls = 0
        info = {"optimal_arm": self._optimal_arm}
        return self._get_obs(), info

    def step(self, action: int):
        self._ensure_not_terminated()
        self._validate_action(action=action)

        self._pulls += 1
        if self._pulls == self._max_steps:
            self._terminated = True
        is_optimal = action == self._optimal_arm
        reward = float(self._np_random.normal(loc=self._arm_means[action], scale=1.0))
        return (
            self._get_obs(),
            reward,
            self._terminated,
            self._truncated,
            self._get_info(optimal_arm_chosen=is_optimal),
        )
