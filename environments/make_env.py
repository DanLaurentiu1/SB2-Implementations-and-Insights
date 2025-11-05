import importlib
from omegaconf import DictConfig


def make_env(env_cfg: DictConfig):
    module_path = env_cfg["module"]
    class_name = env_cfg["class_name"]

    env_mod = importlib.import_module(module_path)
    EnvClass = getattr(env_mod, class_name)

    env = EnvClass()
    return env
