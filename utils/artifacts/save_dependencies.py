from pathlib import Path
import shutil


def save_dependencies(base: Path, file_name: str):
    src = Path("pyproject.toml")
    if src.exists():
        shutil.copy(src, base / file_name)
