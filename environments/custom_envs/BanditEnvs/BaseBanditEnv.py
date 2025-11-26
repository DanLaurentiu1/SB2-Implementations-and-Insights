from abc import ABC, abstractmethod
from gymnasium import Space
import numpy as np


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
    def action_space(self) -> Space:
        pass

    @property
    @abstractmethod
    def observation_space(self) -> Space:
        pass

    @abstractmethod
    def reset(self, *, seed: int = None):
        pass

    @abstractmethod
    def step(self, action: int):
        pass
