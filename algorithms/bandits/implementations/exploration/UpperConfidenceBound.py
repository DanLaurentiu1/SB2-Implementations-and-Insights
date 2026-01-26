import numpy as np
import numpy.typing as npt

from algorithms.bandits.implementations.exploration.ExplorationExploitationContext import (
    ExplorationExploitationContext,
)
from algorithms.bandits.implementations.exploration.ExplorationExploitationStrategy import (
    ExplorationExploitationStrategy,
)
from environments.custom_envs.BanditEnvs.BaseBanditEnv import BaseBanditEnv
from utils.exceptions.logic_exceptions import ExplorationLogicException


class UpperConfidenceBound(ExplorationExploitationStrategy):
    # =================
    # Type Annotations
    # =================

    _uncertainty_coefficient: np.float64
    _action_counts: npt.NDArray[np.float64]

    def __init__(self, uncertainty_coefficient: np.float64, **kwargs):
        self._validate_input(uncertainty_coefficient=float(uncertainty_coefficient))
        self._uncertainty_coefficient = uncertainty_coefficient

    # =================
    # Properties
    # =================

    @property
    def uncertainty_coefficient(self) -> np.float64:
        return self._uncertainty_coefficient

    # =================
    # Public API
    # =================

    def pick_action(self, exploration_context: ExplorationExploitationContext) -> int:
        time_step: int = exploration_context.time_step
        q_values: npt.NDArray[np.float64] = exploration_context.q_values

        zero_indices = np.where(self._action_counts == 0)[0]

        if len(zero_indices) > 0:
            action_picked: int = zero_indices[0]
        else:
            bonuses: npt.NDArray[np.float64] = self._uncertainty_coefficient * np.sqrt(
                np.log(time_step + 1) / self._action_counts
            )
            scores: npt.NDArray[np.float64] = q_values + bonuses

            action_picked: int = int(np.argmax(scores))

        self._action_counts[action_picked] += 1
        return action_picked

    # =================
    # Internals
    # =================

    def _setup(self, env: BaseBanditEnv) -> None:
        self._action_counts = np.zeros(shape=env.number_of_arms, dtype=np.float64)

    def _validate_input(self, uncertainty_coefficient: float) -> None:
        if not (uncertainty_coefficient > 0.0):
            raise ExplorationLogicException(
                f"Invalid uncertainty_coefficient={uncertainty_coefficient}. uncertainty_coefficient must be bigger than 0."
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(c={self._uncertainty_coefficient})"
