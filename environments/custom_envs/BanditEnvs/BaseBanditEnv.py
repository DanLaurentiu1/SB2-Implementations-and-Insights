from abc import ABC, abstractmethod

import numpy as np
from gymnasium import Space

from environments.custom_envs.BanditEnvs.drift.DriftStrategy import DriftStrategy
from environments.custom_envs.BanditEnvs.reward.RewardStrategy import RewardStrategy


class BaseBanditEnv(ABC):
    @property
    @abstractmethod
    def number_of_arms(self) -> int:
        pass

    @property
    @abstractmethod
    def arm_means(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def max_steps(self) -> int:
        pass

    @property
    @abstractmethod
    def seed(self) -> int:
        pass

    @property
    @abstractmethod
    def optimal_arm(self) -> int:
        pass

    @property
    @abstractmethod
    def action_space(self) -> Space:
        pass

    @property
    @abstractmethod
    def observation_space(self) -> Space:
        pass

    @property
    @abstractmethod
    def reward_strategy(self) -> RewardStrategy:
        pass

    @property
    @abstractmethod
    def drift_strategy(self) -> DriftStrategy:
        pass

    @abstractmethod
    def reset(self, *, seed: int = None):
        pass

    @abstractmethod
    def step(
        self, action: int
    ) -> tuple[np.float64, np.float64, bool, bool, dict[str, bool]]:
        pass

    @abstractmethod
    def __str__(self):
        pass
