from datetime import datetime
from pathlib import Path


def make_experiment_directory(base_path: Path, name: str) -> Path:
    date_str = datetime.now().strftime("%d_%B_%Y")
    folder_name = f"{name}_{date_str}"

    processed_experiment_path = base_path / folder_name / "processed"
    processed_experiment_path.mkdir(exist_ok=True, parents=True)

    raw_experiment_path = base_path / folder_name / "raw"
    raw_experiment_path.mkdir(exist_ok=True, parents=True)

    return raw_experiment_path
