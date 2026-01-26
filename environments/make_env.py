from functools import partial
from typing import Callable

import hydra
from omegaconf import DictConfig

from environments.custom_envs.BanditEnvs.BaseBanditEnv import BaseBanditEnv
from environments.custom_envs.BanditEnvs.drift.DriftStrategy import DriftStrategy
from environments.custom_envs.BanditEnvs.reward.RewardStrategy import RewardStrategy


def make_factory(cfg: DictConfig) -> Callable[..., DriftStrategy | RewardStrategy]:
    strategy_class = hydra.utils.get_class(cfg["_target_"])
    params = {str(k): v for k, v in cfg.items() if k != "_target_"}
    strategy_factory = partial(strategy_class, **params)
    return strategy_factory


def make_env(cfg: DictConfig, seed: int) -> BaseBanditEnv:
    env_target = cfg["_target_"]
    env_params = cfg["params"]
    EnvClass = hydra.utils.get_class(env_target)
    drift_factory = make_factory(env_params["drift_strategy"])
    reward_factory = make_factory(env_params["reward_strategy"])
    env = EnvClass(
        number_of_arms=env_params["number_of_arms"],
        max_steps=env_params["max_steps"],
        seed=seed,
        drift_factory=drift_factory,
        reward_factory=reward_factory,
    )
    return env
