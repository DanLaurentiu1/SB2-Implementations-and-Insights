import sys
from pathlib import Path
from omegaconf import DictConfig
from hydra import initialize, compose, main

from algorithms.bandits.run import run
from utils.artifacts.save_git_hash import save_git_hash
from utils.artifacts.save_cli_command import save_cli_command
from utils.artifacts.save_seed import save_seed
from utils.artifacts.save_dependencies import save_dependencies
from utils.artifacts.save_config import save_config

ROOT = Path(__file__).parent.resolve()


@main(config_path="configs", config_name="default")
def run_reproducibility(cfg: DictConfig):
    folder_name = str(cfg["repro_folder_name"])
    reproducibility_dir = ROOT / "reproducibility" / folder_name
    reproducibility_dir.mkdir(exist_ok=True)

    save_git_hash(base=reproducibility_dir, file_name="git_hash.txt")
    save_cli_command(base=reproducibility_dir, file_name="cli_command.txt")
    save_seed(
        agent_seed=cfg["run"]["agent_seed"],
        env_seed=cfg["run"]["env_seed"],
        base=reproducibility_dir,
        file_name="seeds.txt",
    )
    save_dependencies(base=reproducibility_dir, file_name="pyproject.toml")
    save_config(cfg=cfg, base=reproducibility_dir, file_name="config_used.yaml")

    run(cfg)


if __name__ == "__main__":
    run_reproducibility()
