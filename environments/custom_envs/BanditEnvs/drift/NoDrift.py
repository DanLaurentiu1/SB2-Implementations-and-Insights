import numpy as np
import numpy.typing as npt
from environments.custom_envs.BanditEnvs.drift.DriftStrategy import DriftStrategy


class NoDrift(DriftStrategy):
    # =================
    # Public API
    # =================

    def drift(
        self, arm_means: npt.NDArray[np.float64], rng: np.random.Generator
    ) -> npt.NDArray[np.float64]:
        return arm_means

    def reset(self) -> None:
        pass

    # =================
    # Internals
    # =================

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
