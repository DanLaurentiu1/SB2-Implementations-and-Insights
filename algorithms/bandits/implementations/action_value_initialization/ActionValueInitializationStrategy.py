from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
from environments.custom_envs.BanditEnvs.BaseBanditEnv import BaseBanditEnv


class ActionValueInitializationStrategy(ABC):
    def __init__(self, **kwargs):
        pass

    def _setup(self, env: BaseBanditEnv) -> None:
        pass

    @abstractmethod
    def init_action_values(self) -> npt.NDArray[np.float64]:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass
