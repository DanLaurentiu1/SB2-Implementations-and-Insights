from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseLogger(ABC):
    @abstractmethod
    def log(self, row: Dict[str, Any]):
        pass

    @abstractmethod
    def get_path(self):
        pass
