import numpy as np
import numpy.typing as npt
from algorithms.bandits.implementations.action_value_update.ActionValueStrategy import (
    ActionValueUpdateStrategy,
)
from environments.custom_envs.BanditEnvs.BaseBanditEnv import BaseBanditEnv


class AverageSampling(ActionValueUpdateStrategy):
    # =================
    # Type Annotations
    # =================

    _action_counts: npt.NDArray[np.float64]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # =================
    # Public API
    # =================

    def update_action_value(
        self, q_values: npt.NDArray[np.float64], action: int, reward: np.float64
    ) -> None:
        self._action_counts[action] += 1
        q_values[action] += (1 / self._action_counts[action]) * (
            reward - q_values[action]
        )

    # =================
    # Internals
    # =================

    def _setup(self, env: BaseBanditEnv) -> None:
        self._action_counts = np.zeros(shape=env.number_of_arms, dtype=np.float64)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
