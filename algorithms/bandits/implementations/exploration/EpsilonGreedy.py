import numpy as np

from algorithms.bandits.implementations.action_value_update.ActionUpdateContext import (
    ActionUpdateContext,
)
from algorithms.bandits.implementations.exploration.ExplorationExploitationContext import (
    ExplorationExploitationContext,
)
from algorithms.bandits.implementations.exploration.ExplorationExploitationStrategy import (
    ExplorationExploitationStrategy,
)
from utils.exceptions.logic_exceptions import ExplorationLogicException


class EpsilonGreedy(ExplorationExploitationStrategy):
    # =================
    # Type Annotations
    # =================

    _epsilon: np.float64

    def __init__(self, epsilon: np.float64, **kwargs):
        self._validate_input(epsilon=float(epsilon))

        self._epsilon = epsilon

    # =================
    # Properties
    # =================

    @property
    def epsilon(self) -> np.float64:
        return self._epsilon

    # =================
    # Public API
    # =================

    def pick_action(
        self, exploration_context: ExplorationExploitationContext
    ) -> ActionUpdateContext:
        rng = exploration_context.rng
        action_space = exploration_context.action_space
        q_values = exploration_context.q_values

        if rng.random() < self._epsilon:
            action = action_space.sample()
        else:
            action = np.argmax(q_values)

        return ActionUpdateContext(action=int(action))

    # =================
    # Internals
    # =================

    def _validate_input(self, epsilon: float) -> None:
        if not (0.0 <= epsilon <= 1.0):
            raise ExplorationLogicException(
                f"Invalid epsilon={epsilon}. Epsilon must be between 0 and 1."
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(eps={self._epsilon})"
