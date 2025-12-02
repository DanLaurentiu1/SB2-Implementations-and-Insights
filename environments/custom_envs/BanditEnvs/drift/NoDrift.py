import numpy as np
from environments.custom_envs.BanditEnvs.drift.DriftStrategy import DriftStrategy


class NoDrift(DriftStrategy):
    def drift(self, arm_means: np.ndarray) -> np.ndarray:
        # drifts = rng.normal(loc=self._mean, scale=self._variance, size=arm_means.size)
        # return arm_means + drifts
        return arm_means

    def reset(self):
        pass
