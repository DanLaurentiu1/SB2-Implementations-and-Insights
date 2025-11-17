from datetime import datetime
from pathlib import Path


def make_experiment_directory(base_path: Path, name: str) -> Path:
    date_str = datetime.now().strftime("%d_%B_%Y")
    folder_name = f"{name}_{date_str}"

    experiment_path = base_path / folder_name

    experiment_path.mkdir(exist_ok=True)
    return experiment_path
