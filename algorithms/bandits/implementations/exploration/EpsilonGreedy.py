import numpy as np
from gymnasium import Space

from algorithms.bandits.implementations.exploration.ExplorationExploitationStrategy import (
    ExplorationExploitationStrategy,
)
from utils.exceptions.logic_exceptions import ExplorationLogicException


class EpsilonGreedy(ExplorationExploitationStrategy):
    def __init__(self, epsilon: float, **kwargs):
        self._validate_input(epsilon=epsilon)

        self._epsilon = epsilon

    # ==============
    # Properties
    # ==============

    @property
    def epsilon(self) -> float:
        return self._epsilon

    # ==============
    # Public API
    # ==============

    def pick_action(
        self, rng: np.random.Generator, action_space: Space, q_values: np.ndarray
    ) -> int:
        if rng.random() < self._epsilon:
            action = action_space.sample()
        else:
            action = np.argmax(q_values)
        return int(action)

    # ==============
    # Internals
    # ==============

    def _validate_input(self, epsilon: float):
        if not (0.0 <= epsilon <= 1.0):
            raise ExplorationLogicException(
                f"Invalid epsilon={epsilon}. Epsilon must be between 0 and 1."
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(eps={self._epsilon})"
