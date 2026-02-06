from functools import partial
from typing import Callable

import numpy as np
import numpy.typing as npt
from gymnasium import Env, Space
from gymnasium.spaces import Discrete
from gymnasium.utils.seeding import np_random

from environments.custom_envs.BanditEnvs.BaseBanditEnv import BaseBanditEnv
from environments.custom_envs.BanditEnvs.drift.DriftStrategy import DriftStrategy
from environments.custom_envs.BanditEnvs.drift.NoDrift import NoDrift
from environments.custom_envs.BanditEnvs.reward.GaussianReward import GaussianReward
from environments.custom_envs.BanditEnvs.reward.RewardStrategy import RewardStrategy
from utils.exceptions.logic_exceptions import EnvironmentLogicException


class KArmEnvironment(Env, BaseBanditEnv):
    # =================
    # Type Annotations
    # =================

    _terminated: bool
    _truncated: bool
    _number_of_arms: int
    _seed: int
    _pulls: int
    _max_steps: int
    _np_random: np.random.Generator
    _observation_space: Discrete
    _action_space: Discrete
    _drift_strategy: DriftStrategy
    _reward_strategy: RewardStrategy
    _optimal_arm: int
    _arms: npt.NDArray[np.float64]
    _arms_mean: float
    _arms_variance: float

    def __init__(
        self,
        number_of_arms: int = 10,
        seed: int = 16,
        max_steps: int = 1000,
        arms_mean: float = 0.0,
        arms_variance: float = 1.0,
        drift_factory: Callable[..., DriftStrategy] = partial(NoDrift),
        reward_factory: Callable[..., RewardStrategy] = partial(
            GaussianReward, variance=np.float64(1)
        ),
    ):
        self._validate_input(
            number_of_arms=number_of_arms,
            seed=seed,
            max_steps=max_steps,
            arms_mean=arms_mean,
            arms_variance=arms_variance,
        )

        self._terminated: bool = False
        self._truncated: bool = False
        self._number_of_arms: int = number_of_arms
        self._seed: int = seed
        self._pulls: int = 0
        self._max_steps: int = max_steps
        self._arms_mean: float = arms_mean
        self._arms_variance: float = arms_variance
        self._np_random, _ = np_random(self._seed)
        self._observation_space = Discrete(1, seed=self._seed)
        self._action_space = Discrete(n=self._number_of_arms, seed=self._seed, start=0)

        self._drift_strategy = drift_factory()
        self._reward_strategy = reward_factory()

        self._get_new_arms()

    # =================
    # Properties
    # =================

    @property
    def number_of_arms(self) -> int:
        return self._number_of_arms

    @property
    def arms(self) -> np.ndarray:
        return self._arms

    @property
    def max_steps(self) -> int:
        return self._max_steps

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def optimal_arm(self) -> int:
        return self._optimal_arm

    @property
    def action_space(self) -> Space:
        return self._action_space

    @property
    def observation_space(self) -> Space:
        return self._observation_space

    @property
    def reward_strategy(self) -> RewardStrategy:
        return self._reward_strategy

    @property
    def drift_strategy(self) -> DriftStrategy:
        return self._drift_strategy

    # =================
    # Public API
    # =================

    def reset(self, *, seed: int | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
            self._np_random, _ = np_random(self._seed)
            self._get_new_arms()

        self._terminated = False
        self._pulls = 0
        info = {"optimal_arm": self._optimal_arm}
        return self._get_obs(), info

    def step(
        self, action: int
    ) -> tuple[np.float64, np.float64, bool, bool, dict[str, bool]]:
        self._ensure_not_terminated()
        self._validate_action(action=action)

        reward = self._reward_strategy.get_reward(
            arm_mean=self._arms[action], rng=self._np_random
        )

        self._pulls += 1
        if self._pulls == self._max_steps:
            self._terminated = True
        is_optimal = action == self._optimal_arm

        self._arms = self._drift_strategy.drift(
            arm_means=self._arms, rng=self._np_random
        )
        self._optimal_arm = int(np.argmax(self._arms))

        return (
            self._get_obs(),
            reward,
            self._terminated,
            self._truncated,
            self._get_info(optimal_arm_chosen=is_optimal),
        )

    # =================
    # Internals
    # =================

    def _get_new_arms(self):
        self._arms = self._np_random.normal(
            loc=np.float64(self._arms_mean),
            scale=np.float64(self._arms_variance),
            size=self._number_of_arms,
        )
        self._optimal_arm = int(np.argmax(self._arms))

    def _get_obs(self):
        return np.float64(1)

    def _get_info(self, optimal_arm_chosen: bool):
        return {"optimal_arm_chosen": optimal_arm_chosen}

    def _ensure_not_terminated(self):
        if self._terminated:
            raise EnvironmentLogicException(
                "step() called after rollout termination. call reset() first."
            )

    def _validate_input(
        self,
        number_of_arms: int,
        seed: int,
        max_steps: int,
        arms_mean: float,
        arms_variance: float,
    ):
        if number_of_arms < 1:
            raise EnvironmentLogicException(
                f"Invalid number of arms={number_of_arms}. This number must be positive and bigger than 0."
            )
        if max_steps < 1:
            raise EnvironmentLogicException(
                f"Invalid number of max_steps={max_steps}. This number must be positive and bigger than 0."
            )
        if seed < 0:
            raise EnvironmentLogicException(
                f"Invalid seed={seed}. This number must be positive."
            )
        if arms_variance < 0:
            raise EnvironmentLogicException(
                f"Invalid arms_variance={arms_variance}. This number must be positive."
            )

    def _validate_action(self, action: int):
        if not (0 <= action < self._number_of_arms):
            raise EnvironmentLogicException(
                f"Invalid action={action}. Action must be a member of [0, {self._number_of_arms - 1}]"
            )

    # removed reward strategy representation because of windows file length limit (!)
    def __str__(self):
        return f"KArm(s={self._seed},dft={self.drift_strategy.__class__.__name__})"

    def __repr__(self):
        return f"KArm(\n\tdft={self.drift_strategy.__repr__()},\n\tr={self.reward_strategy.__repr__()},\n\ts={self._seed},\n\tarms={self._number_of_arms}\n)"
