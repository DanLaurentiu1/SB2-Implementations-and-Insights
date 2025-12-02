import numpy as np
from environments.custom_envs.BanditEnvs.drift.DriftStrategy import DriftStrategy


class GaussianDrift(DriftStrategy):
    def __init__(self, mean: float = 0, variance: float = 0.01):
        self._mean = mean
        self._variance = variance

    def drift(self, arm_means: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        drifts = rng.normal(loc=self._mean, scale=self._variance, size=arm_means.size)
        return arm_means + drifts

    def reset(self):
        pass
