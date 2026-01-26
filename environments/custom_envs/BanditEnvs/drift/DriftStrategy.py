from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class DriftStrategy(ABC):
    # =================
    # Public API
    # =================

    @abstractmethod
    def drift(
        self, arm_means: npt.NDArray[np.float64], rng: np.random.Generator
    ) -> npt.NDArray[np.float64]:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    # =================
    # Internals
    # =================

    @abstractmethod
    def __repr__(self) -> str:
        pass
