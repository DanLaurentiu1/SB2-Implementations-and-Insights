import numpy as np
from environments.custom_envs.BanditEnvs.drift.DriftStrategy import DriftStrategy


class NoDrift(DriftStrategy):
    def drift(self, arm_means: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return arm_means

    def reset(self):
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
