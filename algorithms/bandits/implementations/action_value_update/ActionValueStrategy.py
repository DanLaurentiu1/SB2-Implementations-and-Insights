from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
from algorithms.bandits.implementations.action_value_update.ActionUpdateContext import (
    ActionUpdateContext,
)
from environments.custom_envs.BanditEnvs.BaseBanditEnv import BaseBanditEnv


class ActionValueUpdateStrategy(ABC):
    def __init__(self, **kwargs):
        pass

    # ==============
    # Public API
    # ==============

    @abstractmethod
    def update_action_value(
        self,
        q_values: npt.NDArray[np.float64],
        reward: np.float64,
        action_update_context: ActionUpdateContext,
    ) -> None:
        pass

    # ==============
    # Internals
    # ==============

    def _setup(self, env: BaseBanditEnv) -> None:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass
