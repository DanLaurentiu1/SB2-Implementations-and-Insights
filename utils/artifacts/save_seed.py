from pathlib import Path


def save_seed(seed: int, base: Path, file_name: str):
    with open(base / file_name, "w") as f:
        f.write(str(seed))
