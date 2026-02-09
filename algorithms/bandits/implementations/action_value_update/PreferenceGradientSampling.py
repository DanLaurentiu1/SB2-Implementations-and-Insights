import numpy as np
import numpy.typing as npt
from algorithms.bandits.implementations.action_value_update.ActionUpdateContext import (
    ActionUpdateContext,
)
from algorithms.bandits.implementations.action_value_update.ActionValueStrategy import (
    ActionValueUpdateStrategy,
)
from utils.exceptions.logic_exceptions import ActionUpdateLogicException


class PreferenceGradientSampling(ActionValueUpdateStrategy):
    # =================
    # Type Annotations
    # =================

    _alpha: np.float64
    _is_baseline: bool
    _baseline: np.float64
    _count: int

    def __init__(self, alpha: np.float64, baseline: bool, **kwargs):
        super().__init__(**kwargs)
        self._validate_input(alpha=float(alpha))

        self._is_baseline = baseline
        self._alpha = alpha
        self._count = 0

        self._init_baseline()

    # ==============
    # Public API
    # ==============

    def update_action_value(
        self,
        q_values: npt.NDArray[np.float64],
        reward: np.float64,
        action_update_context: ActionUpdateContext,
    ) -> None:
        action = action_update_context.action
        if action_update_context.probabilities is None:
            raise ActionUpdateLogicException(
                "PreferenceGradientSampling requires probabilities in the ActionUpdateContext."
            )

        probabilities = action_update_context.probabilities
        error_signal: np.float64 = self._alpha * (reward - self._baseline)

        # ruling out everyone here, we will update the 'chosen action' later
        q_values -= error_signal * probabilities
        q_values[action] += error_signal

        if self._is_baseline:
            self._update_baseline(latest_reward=reward)

    # ==============
    # Internals
    # ==============

    def _update_baseline(self, latest_reward: np.float64) -> None:
        self._count += 1
        self._baseline = self._baseline + (latest_reward - self._baseline) / self._count

    def _init_baseline(self):
        self._baseline = np.float64(0)

    def _validate_input(self, alpha: float) -> None:
        if not (alpha > 0):
            raise ActionUpdateLogicException(
                f"Invalid alpha={alpha}. Alpha must be bigger than 0."
            )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(alpha={self._alpha},baseline={self._baseline})"
        )
