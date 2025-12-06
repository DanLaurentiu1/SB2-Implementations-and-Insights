import numpy as np
from algorithms.bandits.implementations.action_value.ActionValueStrategy import (
    ActionValueStrategy,
)
from environments.custom_envs.BanditEnvs.BaseBanditEnv import BaseBanditEnv


class AverageSampling(ActionValueStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setup(self, env: BaseBanditEnv) -> None:
        self._action_counts = np.zeros(size=env.number_of_arms)

    def update_action_value(
        self, q_values: np.ndarray, action: int, reward: float
    ) -> None:

        self._action_counts[action] += 1
        q_values[action] += (1 / self._action_counts[action]) * (
            reward - q_values[action]
        )
