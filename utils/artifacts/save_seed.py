from pathlib import Path


def save_seed(agent_seed: int, env_seed: int, base: Path, file_name: str):
    file_path = base / file_name
    with file_path.open("a") as f:
        f.write(f"{agent_seed},{env_seed}\n")
