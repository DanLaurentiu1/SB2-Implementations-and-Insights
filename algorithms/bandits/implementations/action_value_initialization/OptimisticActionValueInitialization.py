import numpy as np
import numpy.typing as npt
from algorithms.bandits.implementations.action_value_initialization.ActionValueInitializationStrategy import (
    ActionValueInitializationStrategy,
)
from environments.custom_envs.BanditEnvs.BaseBanditEnv import BaseBanditEnv


class OptimisticActionValueInitialization(ActionValueInitializationStrategy):
    # =================
    # Type Annotations
    # =================

    _number_of_arms: int
    _optimal_arm_mean_value: np.float64
    _env_reward_variance: np.float64
    _q_values: npt.NDArray[np.float64]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # =================
    # Public API
    # =================

    def init_action_values(self) -> npt.NDArray[np.float64]:
        optimistic_value: np.float64 = (
            self._optimal_arm_mean_value + 4 * self._env_reward_variance
        )
        self._q_values = np.full(
            shape=self._number_of_arms,
            fill_value=optimistic_value,
        )
        return self._q_values

    # =================
    # Internals
    # =================

    def _setup(self, env: BaseBanditEnv) -> None:
        self._number_of_arms = env.number_of_arms
        self._optimal_arm_mean_value = env.arm_means[env.optimal_arm]
        self._env_reward_variance = env.reward_strategy.variance

    def __repr__(self) -> str:
        return "OptimisticAVInit"
