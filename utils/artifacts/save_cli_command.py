from pathlib import Path
import sys


def save_cli_command(base: Path, file_name: str):
    argv = sys.argv[1:]
    script_path = Path(sys.argv[0])
    parts = script_path.with_suffix("").parts
    module_name = ".".join(parts[-3:])
    command = f"python -m {module_name} {' '.join(argv)}"
    with open(base / file_name, "w") as f:
        f.write(command)
