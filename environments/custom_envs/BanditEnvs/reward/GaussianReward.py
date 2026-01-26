import numpy as np

from environments.custom_envs.BanditEnvs.reward.RewardStrategy import RewardStrategy
from utils.exceptions.logic_exceptions import RewardLogicException


class GaussianReward(RewardStrategy):
    # =================
    # Type Annotations
    # =================

    _variance: np.float64

    def __init__(self, variance: np.float64):
        self._validate_input(variance=variance)

        self._variance = variance

    # =================
    # Properties
    # =================

    @property
    def variance(self) -> np.float64:
        return self._variance

    # =================
    # Public API
    # =================

    def get_reward(self, arm_mean: np.float64, rng: np.random.Generator) -> np.float64:
        return np.float64(rng.normal(loc=arm_mean, scale=self._variance))

    # =================
    # Internals
    # =================

    def _validate_input(self, variance: np.float64) -> None:
        max_float = np.finfo(np.float64).max
        if variance < 0:
            raise RewardLogicException(
                f"Invalid variance={variance}. Variance cannot be negative."
            )
        if not np.isfinite(variance) or abs(variance) >= max_float:
            raise RewardLogicException(
                f"Invalid variance={variance}. Variance must be finite and within ±{max_float}."
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(var={self._variance})"
