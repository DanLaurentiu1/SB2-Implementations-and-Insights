from abc import ABC, abstractmethod

import numpy as np


class DriftStrategy(ABC):
    @abstractmethod
    def drift(self, arm_means: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        pass

    @abstractmethod
    def reset(self):
        pass
