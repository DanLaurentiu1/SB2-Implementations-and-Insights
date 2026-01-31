import numpy as np
import numpy.typing as npt
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
        self, q_values: npt.NDArray[np.float64], action: int, reward: np.float64
    ) -> None:
        if self._is_baseline:
            # q_values[action] =
            # everything else is going to decrease
            self._update_baseline(latest_reward=reward)
        else:
            pass

    # ==============
    # Internals
    # ==============

    def _update_baseline(self, latest_reward: np.float64) -> None:
        self._count += 1
        self._baseline = self._baseline + (latest_reward - self._baseline) / self._count

    def _init_baseline(self):
        if self._is_baseline:
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
