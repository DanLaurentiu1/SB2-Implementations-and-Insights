from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
from environments.custom_envs.BanditEnvs.BaseBanditEnv import BaseBanditEnv


class ActionValueUpdateStrategy(ABC):
    def __init__(self, **kwargs):
        pass

    def _setup(self, env: BaseBanditEnv) -> None:
        pass

    @abstractmethod
    def update_action_value(
        self, q_values: npt.NDArray[np.float64], action: int, reward: float
    ) -> None:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass
