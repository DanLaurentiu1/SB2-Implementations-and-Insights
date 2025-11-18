import importlib
from omegaconf import DictConfig

from algorithms.bandits.implementations.BaseBanditAgent import BaseBanditAgent
from environments.custom_envs.BanditEnvs.BaseBanditEnv import BaseBanditEnv


def make_agent(env_cfg: DictConfig, seed: int, env: BaseBanditEnv) -> BaseBanditAgent:
    module_path = env_cfg["module"]
    class_name = env_cfg["class_name"]
    params = env_cfg["params"]

    env_mod = importlib.import_module(module_path)
    AgentClass = getattr(env_mod, class_name)

    agent = AgentClass(**params, seed=seed, env=env)
    return agent
