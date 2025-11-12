import numpy as np
from gymnasium.utils.seeding import np_random
from gymnasium import Env
from gymnasium.spaces import Discrete


class KArmEnvironment(Env):
    def __init__(self, number_of_arms: int = 10, seed: int = 16, max_steps: int = 1000):
        self.number_of_arms = number_of_arms
        self.seed = seed
        self.pulls = 0
        self.max_steps = max_steps
        self.np_random, _ = np_random(self.seed)
        self._observation_space = Discrete(1, seed=self.seed)
        self._action_space = Discrete(n=self.number_of_arms, seed=self.seed, start=0)
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

    def reset(self, *, seed=None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed = seed
            self.np_random, _ = np_random(self.seed)
            self._get_new_arms()

        self.pulls = 0
        info = {"optimal_arm": self.optimal_arm}
        return self._get_obs(), info

    def step(self, action: int):
        truncated = False
        self.pulls += 1
        terminated = self.pulls == self.max_steps
        is_optimal = action == self.optimal_arm
        # this works only if action is [0, self.number_of_arms - 1]
        reward = self.np_random.normal(loc=self.arm_means[action], scale=1.0)
        return (
            self._get_obs(),
            reward,
            terminated,
            truncated,
            self._get_info(optimal_arm_chosen=is_optimal),
        )
