from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseLogger(ABC):
    @abstractmethod
    def log(self, row: Dict[str, Any]):
        pass

    @abstractmethod
    def get_path(self):
        pass
