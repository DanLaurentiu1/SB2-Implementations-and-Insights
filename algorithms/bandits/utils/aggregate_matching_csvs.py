import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def aggregate_matching_csvs(input_folder: Path, keyword: str, output_csv: Path):
    memory = np.zeros(shape=(10000 + 1, 3))
    all_csvs: List[Path] = []

    for file_path in input_folder.iterdir():
        if keyword in file_path.name:
            all_csvs.append(file_path)

    print(f"{len(all_csvs)} csvs were found!")
    if not all_csvs:
        raise ValueError(
            f"No CSV files found in {input_folder} containing keyword={keyword}"
        )

    all_csvs.sort()

    for file_path in all_csvs:
        df = pd.read_csv(file_path)
        for row in df.to_numpy():
            step, avg_reward, optimal_chosen = (
                int(row[0]),
                float(row[4]),
                float(row[6]),
            )

            memory[step, 2] += 1
            memory[step, 0] = (
                memory[step, 0] * (memory[step, 2] - 1) + avg_reward
            ) / memory[step, 2]
            memory[step, 1] = (
                memory[step, 1] * (memory[step, 2] - 1) + optimal_chosen
            ) / memory[step, 2]

    steps = np.arange(len(memory))
    df_out = pd.DataFrame(
        {"step": steps, "avg_reward": memory[:, 0], "avg_optimal_chosen": memory[:, 1]}
    )
    df_out.to_csv(output_csv, index=False)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m path.to.file <epsilon>")

    aggregate_matching_csvs(
        input_folder=Path(
            "algorithms/bandits/experiments/Robbin-Monro_08_December_2025/raw"
        ),
        keyword=sys.argv[1],
        output_csv=Path(
            f"algorithms/bandits/experiments/Robbin-Monro_08_December_2025/processed/aggregated_over_{sys.argv[1]}.csv"
        ),
    )
