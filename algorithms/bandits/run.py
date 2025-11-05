from pathlib import Path
import importlib, random, numpy as np
from omegaconf import DictConfig
from algorithms.bandits.utils.make_experiment_directory import make_experiment_directory
from environments.make_env import make_env
from hydra import main

from utils.logging.CSVLogger import CSVLogger

ROOT = Path(__file__).parent.resolve()


def set_global_seed(seed: int):
    print(f"Hit set seed with value={seed}!")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


@main(config_path="configs", config_name="default", version_base=None)
def run(cfg: DictConfig):
    experiments_base = ROOT / "experiments"
    run_name = cfg["run"]["name"]
    experiment_directory = make_experiment_directory(experiments_base, run_name)

    seed = cfg["run"]["seed"]
    if seed is not None:
        set_global_seed(int(seed))

    env = make_env(cfg["environment"])

    module_path = cfg["algorithm"]["module"]
    class_name = cfg["algorithm"]["class_name"]
    impl_mod = importlib.import_module(module_path)
    AgentClass = getattr(impl_mod, class_name)

    agent = AgentClass(
        cfg["algorithm"]["params"]
    )  # bug here, we do not have the actual values, but dictconf
    if hasattr(agent, "seed"):
        agent.seed(int(seed))

    logger = CSVLogger(directory=experiment_directory, columns=agent.get_metrics())

    episodes = int(cfg["run"]["episodes"])
    for ep in range(episodes):
        out = agent.run_episode(env)
        # make my other metrics
        row = {"episode": ep, "episode_reward": out["episode_reward"]}
        logger.log(row)
    print("Run finished. Results in:", experiment_directory)


if __name__ == "__main__":
    run()
