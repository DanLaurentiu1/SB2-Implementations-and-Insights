from functools import partial
from typing import Callable, List

import numpy as np
import numpy.typing as npt
from gymnasium.utils.seeding import np_random

from algorithms.bandits.implementations.action_value_initialization.ActionValueInitializationStrategy import (
    ActionValueInitializationStrategy,
)
from algorithms.bandits.implementations.action_value_initialization.NormalActionValueInitialization import (
    NormalActionValueInitialization,
)
from algorithms.bandits.implementations.action_value_update.ActionValueStrategy import (
    ActionValueUpdateStrategy,
)
from algorithms.bandits.implementations.action_value_update.AverageSampling import (
    AverageSampling,
)
from algorithms.bandits.implementations.BaseBanditAgent import BaseBanditAgent
from algorithms.bandits.implementations.exploration.EpsilonGreedy import EpsilonGreedy
from algorithms.bandits.implementations.exploration.ExplorationExploitationStrategy import (
    ExplorationExploitationStrategy,
)
from environments.custom_envs.BanditEnvs.BaseBanditEnv import BaseBanditEnv
from utils.exceptions.logic_exceptions import AgentLogicException
from utils.logging.BaseLogger import BaseLogger


class BanditAgent(BaseBanditAgent):
    # =================
    # Type Annotations
    # =================

    _env: BaseBanditEnv
    _seed: int
    _metrics: List[str]
    _n_arms: int
    _exploration_strategy: ExplorationExploitationStrategy
    _action_value_update_strategy: ActionValueUpdateStrategy
    _action_value_initialization_strategy: ActionValueInitializationStrategy
    _np_random: np.random.Generator
    _q_values: npt.NDArray[np.float64]

    def __init__(
        self,
        env: BaseBanditEnv,
        seed: int,
        metrics: List[str],
        exploration_factory: Callable[..., ExplorationExploitationStrategy] = partial(
            EpsilonGreedy, epsilon=np.float64(0.1)
        ),
        action_value_update_factory: Callable[..., ActionValueUpdateStrategy] = partial(
            AverageSampling
        ),
        action_value_initialization_factory: Callable[
            ..., ActionValueInitializationStrategy
        ] = partial(NormalActionValueInitialization),
    ):
        self._validate_input(seed=seed, metrics=metrics)

        self._env = env
        self._seed = seed
        self._metrics = metrics
        self._n_arms = self._env.number_of_arms

        self._exploration_strategy = exploration_factory()
        self._exploration_strategy._setup(env=self._env)

        self._action_value_update_strategy = action_value_update_factory()
        self._action_value_update_strategy._setup(env=self._env)

        self._action_value_initialization_strategy = (
            action_value_initialization_factory()
        )
        self._action_value_initialization_strategy._setup(env=self._env)

        self._init_action_values()
        self._reset_rng()

    # =================
    # Properties
    # =================

    @property
    def env(self) -> BaseBanditEnv:
        return self._env

    @property
    def metrics(self) -> List[str]:
        return self._metrics

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def n_arms(self) -> int:
        return self._n_arms

    @property
    def q_values(self) -> npt.NDArray[np.float64]:
        return self._q_values

    # =================
    # Public API
    # =================

    def run_episode(self, logger: BaseLogger, log_every: int = 1) -> None:
        self._env.reset()

        total_reward: np.float64 = np.float64(0)
        optimal_chosen_counter: int = 0
        total_steps: int = 0
        terminated = truncated = False

        while not terminated and not truncated:
            action: int = self._exploration_strategy.pick_action(
                rng=self._np_random,
                action_space=self._env.action_space,
                q_values=self._q_values,
            )

            _, reward, terminated, truncated, info = self._env.step(action=action)
            self._action_value_update_strategy.update_action_value(
                q_values=self._q_values,
                action=action,
                reward=reward,
            )

            total_reward += reward
            total_steps += 1
            optimal_chosen_counter += info["optimal_arm_chosen"]

            if total_steps % log_every == 0 or terminated or truncated:
                row = {
                    "step": total_steps,
                    "action": action,
                    "reward": float(reward),
                    "total_reward": float(total_reward),
                    "average_reward": float(total_reward) / total_steps,
                    "optimal_chosen_counter": optimal_chosen_counter,
                    "optimal_chosen_percentage": (
                        float(optimal_chosen_counter) / total_steps
                        if total_steps
                        else 0.0
                    ),
                }
                logger.log(row=row)

    # =================
    # Internals
    # =================

    def _init_action_values(self) -> None:
        self._q_values = self._action_value_initialization_strategy.init_action_values()

    def _reset_rng(self) -> None:
        self._np_random, _ = np_random(self._seed)

    def _set_seed(self, new_seed: int) -> None:
        self._seed = new_seed
        self._reset_rng()

    def _validate_input(self, seed: int, metrics: List[str]) -> None:
        if seed < 0:
            raise AgentLogicException(
                f"Invalid seed={seed}. This number must be positive."
            )
        if len(metrics) == 0:
            raise AgentLogicException(
                f"Invalid metrics={metrics}. The array should not be empty."
            )

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(s={self._seed},expl={self._exploration_strategy.__class__.__name__},a_v={self._action_value_update_strategy.__repr__()},init={self._action_value_initialization_strategy.__repr__()})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(s={self._seed},\n\tenv={self._env.__repr__()},\n\ta_v={self._action_value_update_strategy.__repr__()},\n\tinit={self._action_value_initialization_strategy.__repr__()},\n\texpl={self._exploration_strategy.__repr__()}\n)"
