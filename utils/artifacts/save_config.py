from pathlib import Path
from omegaconf import DictConfig, OmegaConf


def save_config(cfg: DictConfig, base: Path, file_name: str):
    cfg_path = base / file_name
    with cfg_path.open("w") as f:
        OmegaConf.save(cfg, f.name)
