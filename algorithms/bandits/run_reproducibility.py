import sys
from pathlib import Path
from omegaconf import DictConfig
from hydra import initialize, compose

from algorithms.bandits.run import run
from utils.artifacts.save_git_hash import save_git_hash
from utils.artifacts.save_cli_command import save_cli_command
from utils.artifacts.save_seed import save_seed
from utils.artifacts.save_dependencies import save_dependencies
from utils.artifacts.save_config import save_config


def run_reproducibility(repro_folder_name: str, cfg_name: str = "default"):
    ROOT = Path(__file__).parent.resolve()
    reproducibility_dir = ROOT / "reproducibility" / repro_folder_name
    reproducibility_dir.mkdir()

    with initialize(config_path="configs"):
        cfg: DictConfig = compose(config_name=cfg_name)

    save_git_hash(base=reproducibility_dir, file_name="git_hash.txt")
    save_cli_command(base=reproducibility_dir, file_name="cli_command.txt")
    save_seed(seed=cfg["run"]["seed"], base=reproducibility_dir, file_name="seed.txt")
    save_dependencies(base=reproducibility_dir, file_name="pyproject.toml")
    save_config(cfg=cfg, base=reproducibility_dir, file_name="config_used.yaml")

    run(cfg)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: poetry run python -m algorithms.bandits.run_reproducibility <folder_name> [config_name]"
        )
        sys.exit(1)
    folder_name = sys.argv[1]
    cfg_name = sys.argv[2] if len(sys.argv) > 2 else "default"
    run_reproducibility(folder_name, cfg_name)
