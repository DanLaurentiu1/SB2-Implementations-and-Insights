import numpy as np
import numpy.typing as npt
from algorithms.bandits.implementations.action_value_update.ActionUpdateContext import (
    ActionUpdateContext,
)
from algorithms.bandits.implementations.action_value_update.ActionValueStrategy import (
    ActionValueUpdateStrategy,
)
from utils.exceptions.logic_exceptions import ActionUpdateLogicException


class ERWAverageSampling(ActionValueUpdateStrategy):
    # =================
    # Type Annotations
    # =================

    _alpha: np.float64

    def __init__(self, alpha: np.float64, **kwargs):
        super().__init__(**kwargs)
        self._validate_input(alpha=float(alpha))

        self._alpha = alpha

    # =================
    # Public API
    # =================

    def update_action_value(
        self,
        q_values: npt.NDArray[np.float64],
        reward: np.float64,
        action_update_context: ActionUpdateContext,
    ) -> None:
        action = action_update_context.action

        q_values[action] += self._alpha * (reward - q_values[action])

    # =================
    # Internals
    # =================

    def _validate_input(self, alpha: float):
        if not (0 < alpha <= 1):
            raise ActionUpdateLogicException(
                f"Invalid alpha={alpha}. Alpha must be between 1 and 0."
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(alpha={self._alpha})"
