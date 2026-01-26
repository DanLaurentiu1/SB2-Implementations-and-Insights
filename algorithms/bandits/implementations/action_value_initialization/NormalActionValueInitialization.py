import numpy as np
import numpy.typing as npt
from algorithms.bandits.implementations.action_value_initialization.ActionValueInitializationStrategy import (
    ActionValueInitializationStrategy,
)
from environments.custom_envs.BanditEnvs.BaseBanditEnv import BaseBanditEnv


class NormalActionValueInitialization(ActionValueInitializationStrategy):
    # =================
    # Type Annotations
    # =================

    _number_of_arms: int
    _q_values: npt.NDArray[np.float64]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # =================
    # Public API
    # =================

    def init_action_values(self) -> npt.NDArray[np.float64]:
        self._q_values = np.zeros(shape=self._number_of_arms)
        return self._q_values

    # =================
    # Internals
    # =================

    def _setup(self, env: BaseBanditEnv) -> None:
        self._number_of_arms = env.number_of_arms

    def __repr__(self) -> str:
        return "NormalAVInit"
