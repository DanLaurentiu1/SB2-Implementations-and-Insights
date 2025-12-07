import numpy as np
from algorithms.bandits.implementations.action_value.ActionValueStrategy import (
    ActionValueStrategy,
)
from utils.exceptions.logic_exceptions import ActionValueLogicException


class ERWAverageSampling(ActionValueStrategy):
    def __init__(self, alpha: float, **kwargs):
        super().__init__(**kwargs)
        self._alpha = alpha

    def update_action_value(
        self, q_values: np.ndarray, action: int, reward: float
    ) -> None:
        q_values[action] += self._alpha * (reward - q_values[action])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(alpha={self._alpha})"
