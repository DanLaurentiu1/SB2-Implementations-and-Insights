from abc import ABC, abstractmethod

import numpy as np

from environments.custom_envs.BanditEnvs import BaseBanditEnv


class ActionValueInitializationStrategy(ABC):
    def __init__(self, **kwargs):
        pass

    def _setup(self, env: BaseBanditEnv) -> None:
        pass

    @abstractmethod
    def init_action_values(self) -> np.ndarray:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass
