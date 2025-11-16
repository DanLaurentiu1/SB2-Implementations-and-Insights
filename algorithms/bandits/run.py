from pathlib import Path
import importlib, random, numpy as np
from omegaconf import DictConfig
from algorithms.make_agent import make_agent
from algorithms.bandits.utils.make_experiment_directory import make_experiment_directory
from environments.make_env import make_env
from hydra import main

from utils.logging.CSVLogger import CSVLogger

ROOT = Path(__file__).parent.resolve()


@main(config_path="configs", config_name="default", version_base=None)
def run(cfg: DictConfig):
    experiments_base = ROOT / "experiments"
    run_name = cfg["run"]["name"]
    experiment_directory = make_experiment_directory(experiments_base, run_name)

    agent_seed = cfg["run"]["agent_seeds"]
    env_seed = cfg["run"]["env_seeds"]
    env = make_env(cfg["environment"], seed=env_seed)
    agent = make_agent(cfg["algorithm"], seed=agent_seed, env=env)

    logger = CSVLogger(directory=experiment_directory, columns=agent.get_metrics())

    episodes = int(cfg["run"]["episodes"])
    for _ in range(episodes):
        agent.run_episode(env, logger=logger, log_every=1)
    print("Run finished. Results in:", experiment_directory)


if __name__ == "__main__":
    run()
