from algorithms.bandits.implementations.exploration.ExplorationExploitationContext import (
    ExplorationExploitationContext,
)
from algorithms.bandits.implementations.exploration.ExplorationExploitationStrategy import (
    ExplorationExploitationStrategy,
)
import numpy as np


class SoftmaxSampling(ExplorationExploitationStrategy):
    def __init__(self, **kwargs):
        pass

    # ==============
    # Public API
    # ==============

    def pick_action(self, exploration_context: ExplorationExploitationContext) -> int:
        preferences = exploration_context.q_values
        rng = exploration_context.rng

        exponential = np.exp(preferences)
        exponential_sum = np.sum(exponential)
        pi = exponential / exponential_sum

        action = rng.choice(a=len(pi), p=pi)
        return int(action)

    # ==============
    # Internals
    # ==============

    def __repr__(self) -> str:
        return "SoftmaxSampling()"
