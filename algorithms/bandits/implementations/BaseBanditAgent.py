from abc import ABC, abstractmethod
from typing import List

import numpy as np

from environments.custom_envs.BanditEnvs.BaseBanditEnv import BaseBanditEnv
from utils.logging.BaseLogger import BaseLogger


class BaseBanditAgent(ABC):
    @abstractmethod
    def run_episode(self, logger: BaseLogger, log_every: int) -> None:
        pass

    @abstractmethod
    def __str__(self):
        pass

    @property
    @abstractmethod
    def env(self) -> BaseBanditEnv:
        pass

    @property
    @abstractmethod
    def metrics(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def seed(self) -> int:
        pass

    @property
    @abstractmethod
    def n_arms(self) -> int:
        pass

    @property
    @abstractmethod
    def q_values(self) -> np.ndarray:
        pass
