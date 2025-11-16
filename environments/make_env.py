import importlib
from omegaconf import DictConfig


def make_env(env_cfg: DictConfig, seed: int):
    module_path = env_cfg["module"]
    class_name = env_cfg["class_name"]
    params = env_cfg["params"]

    env_mod = importlib.import_module(module_path)
    EnvClass = getattr(env_mod, class_name)

    env = EnvClass(**params, seed=seed)
    return env
