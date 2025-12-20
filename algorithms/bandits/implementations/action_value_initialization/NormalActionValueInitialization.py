import numpy as np
from algorithms.bandits.implementations.action_value_initialization.ActionValueInitializationStrategy import (
    ActionValueInitializationStrategy,
)
from environments.custom_envs.BanditEnvs.BaseBanditEnv import BaseBanditEnv


class NormalActionValueInitialization(ActionValueInitializationStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _setup(self, env: BaseBanditEnv) -> None:
        self._number_of_arms = env.number_of_arms

    def init_action_values(self) -> np.ndarray:
        self._q_values = np.zeros(shape=self._number_of_arms)
        return self._q_values

    def __repr__(self) -> str:
        return "NormalAVInit"
