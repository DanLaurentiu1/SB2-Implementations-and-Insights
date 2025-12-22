import numpy as np
import numpy.typing as npt
from environments.custom_envs.BanditEnvs.drift.DriftStrategy import DriftStrategy
from utils.exceptions.logic_exceptions import DriftingLogicException


class GaussianDrift(DriftStrategy):
    # =================
    # Type Annotations
    # =================

    _mean: np.float64
    _variance: np.float64

    def __init__(
        self, mean: np.float64 = np.float64(0), variance: np.float64 = np.float64(0.01)
    ):
        self._validate_input(mean=mean, variance=variance)

        self._mean = mean
        self._variance = variance

    # =================
    # Properties
    # =================

    @property
    def mean(self) -> np.float64:
        return self._mean

    @property
    def variance(self) -> np.float64:
        return self._variance

    # =================
    # Public API
    # =================

    def drift(
        self, arm_means: npt.NDArray[np.float64], rng: np.random.Generator
    ) -> npt.NDArray[np.float64]:
        drifts = rng.normal(loc=self._mean, scale=self._variance, size=arm_means.size)
        return arm_means + drifts

    def reset(self) -> None:
        pass

    # =================
    # Internals
    # =================

    def _validate_input(self, mean: np.float64, variance: np.float64) -> None:
        max_float = np.finfo(np.float64).max
        if variance < 0:
            raise DriftingLogicException(
                f"Invalid variance={variance}. Variance cannot be negative."
            )
        if not np.isfinite(variance) or abs(variance) >= max_float:
            raise DriftingLogicException(
                f"Invalid variance={variance}. Variance must be finite and within ±{max_float}."
            )
        if not np.isfinite(mean) or abs(mean) >= max_float:
            raise DriftingLogicException(
                f"Invalid mean={mean}. Mean must be finite and within ±{max_float}."
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self._mean},var={self._variance})"
