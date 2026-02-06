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
import numpy.typing as npt


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

        exponential: npt.NDArray[np.float64] = np.exp(preferences)
        exponential_sum = np.sum(exponential)
        pi = exponential / exponential_sum

        action = rng.choice(a=len(pi), p=pi)
        return ActionUpdateContext(action=int(action), probabilities=pi)

    # ==============
    # Internals
    # ==============

    def __repr__(self) -> str:
        return "SoftmaxSampling()"
