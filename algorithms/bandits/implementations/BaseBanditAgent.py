from abc import ABC, abstractmethod
from typing import List

from utils.logging.BaseLogger import BaseLogger


class BaseBanditAgent(ABC):
    @abstractmethod
    def run_episode(logger: BaseLogger, log_every: int) -> None:
        pass

    @abstractmethod
    def get_metrics() -> List[str]:
        pass
