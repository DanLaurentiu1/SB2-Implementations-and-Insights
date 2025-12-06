from functools import partial
import importlib
from typing import Callable
import hydra
from omegaconf import DictConfig

from algorithms.bandits.implementations.BaseBanditAgent import BaseBanditAgent
from algorithms.bandits.implementations.action_value.ActionValueStrategy import (
    ActionValueStrategy,
)
from environments.custom_envs.BanditEnvs.BaseBanditEnv import BaseBanditEnv


def make_factory(cfg: DictConfig) -> Callable[..., ActionValueStrategy]:
    strategy_class = hydra.utils.get_class(cfg["_target_"])
    params = {k: v for k, v in cfg.items() if k != "_target_"}
    strategy_factory = partial(strategy_class, **params)
    return strategy_factory


def make_agent(cfg: DictConfig, seed: int, env: BaseBanditEnv) -> BaseBanditAgent:
    agent_target = cfg["_target_"]
    AgentClass = hydra.utils.get_class(agent_target)
    agent_params = cfg["params"]
    action_factory = make_factory(agent_params["action_value_strategy"])
    exploration_factory = make_factory(agent_params["exploration_strategy"])
    metrics = agent_params["metrics"]
    agent = AgentClass(
        env=env,
        seed=seed,
        metrics=metrics,
        action_value_factory=action_factory,
        exploration_factory=exploration_factory,
    )
    return agent
