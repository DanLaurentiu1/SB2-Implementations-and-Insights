from pathlib import Path

from hydra import main
from omegaconf import DictConfig

from algorithms.bandits.utils.make_experiment_directory import make_experiment_directory
from algorithms.make_agent import make_agent
from environments.make_env import make_env
from utils.logging.CSVLogger import CSVLogger

ROOT = Path(__file__).parent.resolve()
EXPERIMENTS_BASE = ROOT / "experiments"


@main(config_path="configs", config_name="default", version_base=None)
def run(cfg: DictConfig):
    run_name = cfg["run"]["name"]
    experiment_directory = make_experiment_directory(EXPERIMENTS_BASE, run_name)

    agent_seed = int(cfg["run"]["agent_seed"])
    env_seed = int(cfg["run"]["env_seed"])
    env = make_env(cfg["environment"], seed=env_seed)
    agent = make_agent(cfg["algorithm"], seed=agent_seed, env=env)

    logger = CSVLogger(
        directory=experiment_directory,
        columns=agent.metrics,
        filename=f"{agent}_{env}_results.csv",
    )

    episodes = int(cfg["run"]["episodes"])
    for _ in range(episodes):
        agent.run_episode(logger=logger, log_every=1)

    print(f"Run finished. Results in: {experiment_directory}\n")


if __name__ == "__main__":
    run()
