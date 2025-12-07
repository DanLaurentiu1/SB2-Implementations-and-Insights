import subprocess
from pathlib import Path


def save_git_hash(base: Path, file_name: str):
    try:
        git_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        )
    except Exception:
        git_hash = "N/A"

    with open(base / file_name, "w") as f:
        f.write(git_hash)
