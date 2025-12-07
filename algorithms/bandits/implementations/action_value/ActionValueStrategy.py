from abc import ABC, abstractmethod

import numpy as np

from environments.custom_envs.BanditEnvs import BaseBanditEnv


class ActionValueStrategy(ABC):
    def __init__(self, **kwargs):
        pass

    def _setup(self, env: BaseBanditEnv) -> None:
        pass

    @abstractmethod
    def update_action_value(
        self, q_values: np.ndarray, action: int, reward: float
    ) -> None:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass
