from algorithms.bandits.implementations.action_value_update.ActionUpdateContext import (
    ActionUpdateContext,
)
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

    def pick_action(
        self, exploration_context: ExplorationExploitationContext
    ) -> ActionUpdateContext:
        preferences = exploration_context.q_values
        rng = exploration_context.rng

        shifted_preferences = preferences - np.max(preferences)
        exponential = np.exp(shifted_preferences)
        pi = exponential / np.sum(exponential)

        action = rng.choice(a=len(pi), p=pi)
        return ActionUpdateContext(action=int(action), probabilities=pi)

    # ==============
    # Internals
    # ==============

    def __repr__(self) -> str:
        return "SoftmaxSampling()"
