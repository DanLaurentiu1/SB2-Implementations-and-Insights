import numpy as np


class GaussianReward:
    def __init__(self, variance: np.float64 = 1.0):
        self._variance = variance

    def get_reward(self, arm_mean: np.float64, rng: np.random.Generator) -> np.float64:
        return rng.normal(loc=arm_mean, scale=self._variance)
