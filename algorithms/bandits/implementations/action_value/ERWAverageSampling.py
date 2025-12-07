import numpy as np

from algorithms.bandits.implementations.action_value.ActionValueStrategy import (
    ActionValueStrategy,
)
from utils.exceptions.logic_exceptions import ActionValueLogicException


class ERWAverageSampling(ActionValueStrategy):
    def __init__(self, alpha: float, **kwargs):
        super().__init__(**kwargs)
        self._validate_input(alpha=alpha)

        self._alpha = alpha

    # ==============
    # Public API
    # ==============

    def update_action_value(
        self, q_values: np.ndarray, action: int, reward: float
    ) -> None:
        q_values[action] += self._alpha * (reward - q_values[action])

    # ==============
    # Internals
    # ==============

    def _validate_input(self, alpha: float):
        if not (0 < alpha <= 1):
            raise ActionValueLogicException(
                f"Invalid alpha={alpha}. Alpha must be between 1 and 0."
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(alpha={self._alpha})"
