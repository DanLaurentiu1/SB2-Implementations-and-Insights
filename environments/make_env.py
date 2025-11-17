import importlib
from omegaconf import DictConfig


def make_env(env_cfg: DictConfig, seed: int):
    module_path = env_cfg["module"]
    class_name = env_cfg["class_name"]
    params = dict(env_cfg["params"])

    env_module = importlib.import_module(module_path)
    env_class = getattr(env_module, class_name)

    env = env_class(**params, seed=seed)
    return env
