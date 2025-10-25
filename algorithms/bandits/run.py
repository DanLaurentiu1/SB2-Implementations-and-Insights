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
    # build a run directory inside experiments/
    experiments_base = ROOT / "experiments"
    run_name = cfg["run"]["name"]
    experiment_directory = "HEHE, directory"
    make_experiment_directory(str(experiments_base), run_name)

    # set the seed
    seed = cfg["run"]["seed"]
    if seed is not None:
        set_global_seed(int(seed))

    # build the environment
    env = make_env(cfg["environment"])

    # dynamic import of Agent class defined by the algo config
    module_path = cfg["algorithm"]["module"]
    class_name = cfg["algorithm"]["class_name"]
    impl_mod = importlib.import_module(module_path)
    AgentClass = getattr(impl_mod, class_name)

    # instantiate the agent
    agent = AgentClass(cfg["algorithm"]["params"])
    if hasattr(agent, "seed"):
        agent.seed(int(seed))

    # create the logger
    logger = CSVLogger(experiment_directory)

    # training loop
    episodes = int(cfg["run"]["episodes"])
    for _ in range(episodes):
        out = agent.run_episode(env)
        print(out)
    print("Run finished. Results in:", experiment_directory)


if __name__ == "__main__":
    run()
