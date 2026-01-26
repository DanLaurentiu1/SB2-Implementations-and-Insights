import shutil
from pathlib import Path


def save_dependencies(base: Path, file_name: str):
    src = Path("pyproject.toml")
    if src.exists():
        shutil.copy(src, base / file_name)
