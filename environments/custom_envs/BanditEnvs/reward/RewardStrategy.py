from abc import ABC, abstractmethod

import numpy as np


class RewardStrategy(ABC):
    # =================
    # Public API
    # =================

    @abstractmethod
    def get_reward(self, arm_mean: np.float64, rng: np.random.Generator) -> np.float64:
        pass

    @property
    @abstractmethod
    def variance(self) -> np.float64:
        pass

    # =================
    # Internals
    # =================

    @abstractmethod
    def __repr__(self) -> str:
        pass
